from __future__ import (absolute_import, division, print_function)
import fnmatch
import os
import sys
import re
import itertools
import traceback
from operator import attrgetter
from random import shuffle
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleOptionsError, AnsibleParserError
from ansible.inventory.data import InventoryData
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.parsing.utils.addresses import parse_address
from ansible.plugins.loader import inventory_loader
from ansible.utils.helpers import deduplicate_list
from ansible.utils.path import unfrackpath
from ansible.utils.display import Display
from ansible.utils.vars import combine_vars
from ansible.vars.plugins import get_vars_from_inventory_sources
class InventoryManager(object):
    """ Creates and manages inventory """

    def __init__(self, loader, sources=None, parse=True, cache=True):
        self._loader = loader
        self._inventory = InventoryData()
        self._restriction = None
        self._subset = None
        self._hosts_patterns_cache = {}
        self._pattern_cache = {}
        if sources is None:
            self._sources = []
        elif isinstance(sources, string_types):
            self._sources = [sources]
        else:
            self._sources = sources
        if parse:
            self.parse_sources(cache=cache)
        self._cached_dynamic_hosts = []
        self._cached_dynamic_grouping = []

    @property
    def localhost(self):
        return self._inventory.get_host('localhost')

    @property
    def groups(self):
        return self._inventory.groups

    @property
    def hosts(self):
        return self._inventory.hosts

    def add_host(self, host, group=None, port=None):
        return self._inventory.add_host(host, group, port)

    def add_group(self, group):
        return self._inventory.add_group(group)

    def get_groups_dict(self):
        return self._inventory.get_groups_dict()

    def reconcile_inventory(self):
        self.clear_caches()
        return self._inventory.reconcile_inventory()

    def get_host(self, hostname):
        return self._inventory.get_host(hostname)

    def _fetch_inventory_plugins(self):
        """ sets up loaded inventory plugins for usage """
        display.vvvv('setting up inventory plugins')
        plugins = []
        for name in C.INVENTORY_ENABLED:
            plugin = inventory_loader.get(name)
            if plugin:
                plugins.append(plugin)
            else:
                display.warning('Failed to load inventory plugin, skipping %s' % name)
        if not plugins:
            raise AnsibleError('No inventory plugins available to generate inventory, make sure you have at least one enabled.')
        return plugins

    def parse_sources(self, cache=False):
        """ iterate over inventory sources and parse each one to populate it"""
        parsed = False
        for source in self._sources:
            if source:
                if ',' not in source:
                    source = unfrackpath(source, follow=False)
                parse = self.parse_source(source, cache=cache)
                if parse and (not parsed):
                    parsed = True
        if parsed:
            self._inventory.reconcile_inventory()
        elif C.INVENTORY_UNPARSED_IS_FAILED:
            raise AnsibleError('No inventory was parsed, please check your configuration and options.')
        elif C.INVENTORY_UNPARSED_WARNING:
            display.warning('No inventory was parsed, only implicit localhost is available')
        for group in self.groups.values():
            group.vars = combine_vars(group.vars, get_vars_from_inventory_sources(self._loader, self._sources, [group], 'inventory'))
        for host in self.hosts.values():
            host.vars = combine_vars(host.vars, get_vars_from_inventory_sources(self._loader, self._sources, [host], 'inventory'))

    def parse_source(self, source, cache=False):
        """ Generate or update inventory for the source provided """
        parsed = False
        failures = []
        display.debug(u'Examining possible inventory source: %s' % source)
        b_source = to_bytes(source)
        if os.path.isdir(b_source):
            display.debug(u'Searching for inventory files in directory: %s' % source)
            for i in sorted(os.listdir(b_source)):
                display.debug(u'Considering %s' % i)
                if IGNORED.search(i):
                    continue
                fullpath = to_text(os.path.join(b_source, i), errors='surrogate_or_strict')
                parsed_this_one = self.parse_source(fullpath, cache=cache)
                display.debug(u'parsed %s as %s' % (fullpath, parsed_this_one))
                if not parsed:
                    parsed = parsed_this_one
        else:
            self._inventory.current_source = source
            for plugin in self._fetch_inventory_plugins():
                plugin_name = to_text(getattr(plugin, '_load_name', getattr(plugin, '_original_path', '')))
                display.debug(u'Attempting to use plugin %s (%s)' % (plugin_name, plugin._original_path))
                try:
                    plugin_wants = bool(plugin.verify_file(source))
                except Exception:
                    plugin_wants = False
                if plugin_wants:
                    try:
                        plugin.parse(self._inventory, self._loader, source, cache=cache)
                        try:
                            plugin.update_cache_if_changed()
                        except AttributeError:
                            pass
                        parsed = True
                        display.vvv('Parsed %s inventory source with %s plugin' % (source, plugin_name))
                        break
                    except AnsibleParserError as e:
                        display.debug('%s was not parsable by %s' % (source, plugin_name))
                        tb = ''.join(traceback.format_tb(sys.exc_info()[2]))
                        failures.append({'src': source, 'plugin': plugin_name, 'exc': e, 'tb': tb})
                    except Exception as e:
                        display.debug('%s failed while attempting to parse %s' % (plugin_name, source))
                        tb = ''.join(traceback.format_tb(sys.exc_info()[2]))
                        failures.append({'src': source, 'plugin': plugin_name, 'exc': AnsibleError(e), 'tb': tb})
                else:
                    display.vvv('%s declined parsing %s as it did not pass its verify_file() method' % (plugin_name, source))
        if parsed:
            self._inventory.processed_sources.append(self._inventory.current_source)
        elif source != '/etc/ansible/hosts' or os.path.exists(source):
            if failures:
                for fail in failures:
                    display.warning(u'\n* Failed to parse %s with %s plugin: %s' % (to_text(fail['src']), fail['plugin'], to_text(fail['exc'])))
                    if 'tb' in fail:
                        display.vvv(to_text(fail['tb']))
            if C.INVENTORY_ANY_UNPARSED_IS_FAILED:
                raise AnsibleError(u'Completely failed to parse inventory source %s' % source)
            else:
                display.warning('Unable to parse %s as an inventory source' % source)
        self._inventory.current_source = None
        return parsed

    def clear_caches(self):
        """ clear all caches """
        self._hosts_patterns_cache = {}
        self._pattern_cache = {}

    def refresh_inventory(self):
        """ recalculate inventory """
        self.clear_caches()
        self._inventory = InventoryData()
        self.parse_sources(cache=False)
        for host in self._cached_dynamic_hosts:
            self.add_dynamic_host(host, {'refresh': True})
        for host, result in self._cached_dynamic_grouping:
            result['refresh'] = True
            self.add_dynamic_group(host, result)

    def _match_list(self, items, pattern_str):
        try:
            if not pattern_str[0] == '~':
                pattern = re.compile(fnmatch.translate(pattern_str))
            else:
                pattern = re.compile(pattern_str[1:])
        except Exception:
            raise AnsibleError('Invalid host list pattern: %s' % pattern_str)
        results = []
        for item in items:
            if pattern.match(item):
                results.append(item)
        return results

    def get_hosts(self, pattern='all', ignore_limits=False, ignore_restrictions=False, order=None):
        """
        Takes a pattern or list of patterns and returns a list of matching
        inventory host names, taking into account any active restrictions
        or applied subsets
        """
        hosts = []
        if isinstance(pattern, list):
            pattern_list = pattern[:]
        else:
            pattern_list = [pattern]
        if pattern_list:
            if not ignore_limits and self._subset:
                pattern_list.extend(self._subset)
            if not ignore_restrictions and self._restriction:
                pattern_list.extend(self._restriction)
            pattern_hash = tuple(pattern_list)
            if pattern_hash not in self._hosts_patterns_cache:
                patterns = split_host_pattern(pattern)
                hosts = self._evaluate_patterns(patterns)
                if not ignore_limits and self._subset:
                    subset_uuids = set((s._uuid for s in self._evaluate_patterns(self._subset)))
                    hosts = [h for h in hosts if h._uuid in subset_uuids]
                if not ignore_restrictions and self._restriction:
                    hosts = [h for h in hosts if h.name in self._restriction]
                self._hosts_patterns_cache[pattern_hash] = deduplicate_list(hosts)
            if order in ['sorted', 'reverse_sorted']:
                hosts = sorted(self._hosts_patterns_cache[pattern_hash][:], key=attrgetter('name'), reverse=order == 'reverse_sorted')
            elif order == 'reverse_inventory':
                hosts = self._hosts_patterns_cache[pattern_hash][::-1]
            else:
                hosts = self._hosts_patterns_cache[pattern_hash][:]
                if order == 'shuffle':
                    shuffle(hosts)
                elif order not in [None, 'inventory']:
                    raise AnsibleOptionsError("Invalid 'order' specified for inventory hosts: %s" % order)
        return hosts

    def _evaluate_patterns(self, patterns):
        """
        Takes a list of patterns and returns a list of matching host names,
        taking into account any negative and intersection patterns.
        """
        patterns = order_patterns(patterns)
        hosts = []
        for p in patterns:
            if p in self._inventory.hosts:
                hosts.append(self._inventory.get_host(p))
            else:
                that = self._match_one_pattern(p)
                if p[0] == '!':
                    that = set(that)
                    hosts = [h for h in hosts if h not in that]
                elif p[0] == '&':
                    that = set(that)
                    hosts = [h for h in hosts if h in that]
                else:
                    existing_hosts = set((y.name for y in hosts))
                    hosts.extend([h for h in that if h.name not in existing_hosts])
        return hosts

    def _match_one_pattern(self, pattern):
        """
        Takes a single pattern and returns a list of matching host names.
        Ignores intersection (&) and exclusion (!) specifiers.

        The pattern may be:

            1. A regex starting with ~, e.g. '~[abc]*'
            2. A shell glob pattern with ?/*/[chars]/[!chars], e.g. 'foo*'
            3. An ordinary word that matches itself only, e.g. 'foo'

        The pattern is matched using the following rules:

            1. If it's 'all', it matches all hosts in all groups.
            2. Otherwise, for each known group name:
                (a) if it matches the group name, the results include all hosts
                    in the group or any of its children.
                (b) otherwise, if it matches any hosts in the group, the results
                    include the matching hosts.

        This means that 'foo*' may match one or more groups (thus including all
        hosts therein) but also hosts in other groups.

        The built-in groups 'all' and 'ungrouped' are special. No pattern can
        match these group names (though 'all' behaves as though it matches, as
        described above). The word 'ungrouped' can match a host of that name,
        and patterns like 'ungr*' and 'al*' can match either hosts or groups
        other than all and ungrouped.

        If the pattern matches one or more group names according to these rules,
        it may have an optional range suffix to select a subset of the results.
        This is allowed only if the pattern is not a regex, i.e. '~foo[1]' does
        not work (the [1] is interpreted as part of the regex), but 'foo*[1]'
        would work if 'foo*' matched the name of one or more groups.

        Duplicate matches are always eliminated from the results.
        """
        if pattern[0] in ('&', '!'):
            pattern = pattern[1:]
        if pattern not in self._pattern_cache:
            expr, slice = self._split_subscript(pattern)
            hosts = self._enumerate_matches(expr)
            try:
                hosts = self._apply_subscript(hosts, slice)
            except IndexError:
                raise AnsibleError("No hosts matched the subscripted pattern '%s'" % pattern)
            self._pattern_cache[pattern] = hosts
        return self._pattern_cache[pattern]

    def _split_subscript(self, pattern):
        """
        Takes a pattern, checks if it has a subscript, and returns the pattern
        without the subscript and a (start,end) tuple representing the given
        subscript (or None if there is no subscript).

        Validates that the subscript is in the right syntax, but doesn't make
        sure the actual indices make sense in context.
        """
        if pattern[0] == '~':
            return (pattern, None)
        subscript = None
        m = PATTERN_WITH_SUBSCRIPT.match(pattern)
        if m:
            pattern, idx, start, sep, end = m.groups()
            if idx:
                subscript = (int(idx), None)
            else:
                if not end:
                    end = -1
                subscript = (int(start), int(end))
                if sep == '-':
                    display.warning('Use [x:y] inclusive subscripts instead of [x-y] which has been removed')
        return (pattern, subscript)

    def _apply_subscript(self, hosts, subscript):
        """
        Takes a list of hosts and a (start,end) tuple and returns the subset of
        hosts based on the subscript (which may be None to return all hosts).
        """
        if not hosts or not subscript:
            return hosts
        start, end = subscript
        if end:
            if end == -1:
                end = len(hosts) - 1
            return hosts[start:end + 1]
        else:
            return [hosts[start]]

    def _enumerate_matches(self, pattern):
        """
        Returns a list of host names matching the given pattern according to the
        rules explained above in _match_one_pattern.
        """
        results = []
        matching_groups = self._match_list(self._inventory.groups, pattern)
        if matching_groups:
            for groupname in matching_groups:
                results.extend(self._inventory.groups[groupname].get_hosts())
        if not matching_groups or pattern[0] == '~' or any((special in pattern for special in ('.', '?', '*', '['))):
            matching_hosts = self._match_list(self._inventory.hosts, pattern)
            if matching_hosts:
                for hostname in matching_hosts:
                    results.append(self._inventory.hosts[hostname])
        if not results and pattern in C.LOCALHOST:
            implicit = self._inventory.get_host(pattern)
            if implicit:
                results.append(implicit)
        if not results and (not matching_groups) and (pattern != 'all'):
            msg = 'Could not match supplied host pattern, ignoring: %s' % pattern
            display.debug(msg)
            if C.HOST_PATTERN_MISMATCH == 'warning':
                display.warning(msg)
            elif C.HOST_PATTERN_MISMATCH == 'error':
                raise AnsibleError(msg)
        return results

    def list_hosts(self, pattern='all'):
        """ return a list of hostnames for a pattern """
        result = self.get_hosts(pattern)
        if len(result) == 0 and pattern in C.LOCALHOST:
            result = [pattern]
        return result

    def list_groups(self):
        return sorted(self._inventory.groups.keys())

    def restrict_to_hosts(self, restriction):
        """
        Restrict list operations to the hosts given in restriction.  This is used
        to batch serial operations in main playbook code, don't use this for other
        reasons.
        """
        if restriction is None:
            return
        elif not isinstance(restriction, list):
            restriction = [restriction]
        self._restriction = set((to_text(h.name) for h in restriction))

    def subset(self, subset_pattern):
        """
        Limits inventory results to a subset of inventory that matches a given
        pattern, such as to select a given geographic of numeric slice amongst
        a previous 'hosts' selection that only select roles, or vice versa.
        Corresponds to --limit parameter to ansible-playbook
        """
        if subset_pattern is None:
            self._subset = None
        else:
            subset_patterns = split_host_pattern(subset_pattern)
            results = []
            for x in subset_patterns:
                if not x:
                    continue
                if x[0] == '@':
                    b_limit_file = to_bytes(x[1:])
                    if not os.path.exists(b_limit_file):
                        raise AnsibleError(u'Unable to find limit file %s' % b_limit_file)
                    if not os.path.isfile(b_limit_file):
                        raise AnsibleError(u'Limit starting with "@" must be a file, not a directory: %s' % b_limit_file)
                    with open(b_limit_file) as fd:
                        results.extend([to_text(l.strip()) for l in fd.read().split('\n')])
                else:
                    results.append(to_text(x))
            self._subset = results

    def remove_restriction(self):
        """ Do not restrict list operations """
        self._restriction = None

    def clear_pattern_cache(self):
        self._pattern_cache = {}

    def add_dynamic_host(self, host_info, result_item):
        """
        Helper function to add a new host to inventory based on a task result.
        """
        changed = False
        if not result_item.get('refresh'):
            self._cached_dynamic_hosts.append(host_info)
        if host_info:
            host_name = host_info.get('host_name')
            if host_name not in self.hosts:
                self.add_host(host_name, 'all')
                changed = True
            new_host = self.hosts.get(host_name)
            new_host_vars = new_host.get_vars()
            new_host_combined_vars = combine_vars(new_host_vars, host_info.get('host_vars', dict()))
            if new_host_vars != new_host_combined_vars:
                new_host.vars = new_host_combined_vars
                changed = True
            new_groups = host_info.get('groups', [])
            for group_name in new_groups:
                if group_name not in self.groups:
                    group_name = self._inventory.add_group(group_name)
                    changed = True
                new_group = self.groups[group_name]
                if new_group.add_host(self.hosts[host_name]):
                    changed = True
            if changed:
                self.reconcile_inventory()
            result_item['changed'] = changed

    def add_dynamic_group(self, host, result_item):
        """
        Helper function to add a group (if it does not exist), and to assign the
        specified host to that group.
        """
        changed = False
        if not result_item.get('refresh'):
            self._cached_dynamic_grouping.append((host, result_item))
        real_host = self.hosts.get(host.name)
        if real_host is None:
            if host.name == self.localhost.name:
                real_host = self.localhost
            elif not result_item.get('refresh'):
                raise AnsibleError('%s cannot be matched in inventory' % host.name)
            else:
                return
        group_name = result_item.get('add_group')
        parent_group_names = result_item.get('parent_groups', [])
        if group_name not in self.groups:
            group_name = self.add_group(group_name)
        for name in parent_group_names:
            if name not in self.groups:
                self.add_group(name)
                changed = True
        group = self._inventory.groups[group_name]
        for parent_group_name in parent_group_names:
            parent_group = self.groups[parent_group_name]
            new = parent_group.add_child_group(group)
            if new and (not changed):
                changed = True
        if real_host not in group.get_hosts():
            changed = group.add_host(real_host)
        if group not in real_host.get_groups():
            changed = real_host.add_group(group)
        if changed:
            self.reconcile_inventory()
        result_item['changed'] = changed