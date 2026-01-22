import logging
import re
from collections import namedtuple
from datetime import time
from urllib.parse import ParseResult, quote, urlparse, urlunparse
def _parse_robotstxt(self, content):
    lines = content.splitlines()
    current_rule_sets = []
    previous_rule_field = None
    for line in lines:
        self._total_line_seen += 1
        hash_pos = line.find('#')
        if hash_pos != -1:
            line = line[0:hash_pos].strip()
        line = line.strip()
        if not line:
            continue
        if line.find(':') != -1:
            field, value = line.split(':', 1)
        else:
            parts = line.split(' ')
            if len(parts) < 2:
                continue
            possible_filed = parts[0]
            for i in range(1, len(parts)):
                if _is_valid_directive_field(possible_filed):
                    field, value = (possible_filed, ' '.join(parts[i:]))
                    break
                possible_filed += ' ' + parts[i]
            else:
                continue
        field = field.strip().lower()
        value = value.strip()
        if not value:
            previous_rule_field = field
            continue
        if not current_rule_sets and field not in _USER_AGENT_DIRECTIVE and (field not in _SITEMAP_DIRECTIVE):
            logger.debug('Rule at line {line_seen} without any user agent to enforce it on.'.format(line_seen=self._total_line_seen))
            continue
        self._total_directive_seen += 1
        if field in _USER_AGENT_DIRECTIVE:
            if previous_rule_field and previous_rule_field not in _USER_AGENT_DIRECTIVE:
                current_rule_sets = []
            user_agent = value.strip().lower()
            user_agent_without_asterisk = None
            if user_agent != '*' and '*' in user_agent:
                user_agent_without_asterisk = user_agent.replace('*', '')
            user_agents = [user_agent, user_agent_without_asterisk]
            for user_agent in user_agents:
                if not user_agent:
                    continue
                rule_set = self._user_agents.get(user_agent, None)
                if rule_set and rule_set not in current_rule_sets:
                    current_rule_sets.append(rule_set)
                if not rule_set:
                    rule_set = _RuleSet(self)
                    rule_set.user_agent = user_agent
                    self._user_agents[user_agent] = rule_set
                    current_rule_sets.append(rule_set)
        elif field in _ALLOW_DIRECTIVE:
            for rule_set in current_rule_sets:
                rule_set.allow(value)
        elif field in _DISALLOW_DIRECTIVE:
            for rule_set in current_rule_sets:
                rule_set.disallow(value)
        elif field in _SITEMAP_DIRECTIVE:
            self._sitemap_list.append(value)
        elif field in _CRAWL_DELAY_DIRECTIVE:
            for rule_set in current_rule_sets:
                rule_set.crawl_delay = value
        elif field in _REQUEST_RATE_DIRECTIVE:
            for rule_set in current_rule_sets:
                rule_set.request_rate = value
        elif field in _HOST_DIRECTIVE:
            self._host = value
        elif field in _VISIT_TIME_DIRECTIVE:
            for rule_set in current_rule_sets:
                rule_set.visit_time = value
        else:
            self._invalid_directive_seen += 1
        previous_rule_field = field
    for user_agent in self._user_agents.values():
        user_agent.finalize_rules()