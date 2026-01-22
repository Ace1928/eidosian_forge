import atexit
import codecs
import contextlib
import copy
import difflib
import doctest
import errno
import functools
import itertools
import logging
import math
import os
import platform
import pprint
import random
import re
import shlex
import site
import stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import unittest
import warnings
from io import BytesIO, StringIO, TextIOWrapper
from typing import Callable, Set
import testtools
from testtools import content
import breezy
from breezy.bzr import chk_map
from .. import branchbuilder
from .. import commands as _mod_commands
from .. import config, controldir, debug, errors, hooks, i18n
from .. import lock as _mod_lock
from .. import lockdir, osutils
from .. import plugin as _mod_plugin
from .. import pyutils, registry, symbol_versioning, trace
from .. import transport as _mod_transport
from .. import ui, urlutils, workingtree
from ..bzr.smart import client, request
from ..tests import TestUtil, fixtures, test_server, treeshape, ui_testing
from ..transport import memory, pathfilter
from testtools.testcase import TestSkipped
def _test_suite_testmod_names():
    """Return the standard list of test module names to test."""
    return ['breezy.bzr.tests', 'breezy.git.tests', 'breezy.tests.blackbox', 'breezy.tests.commands', 'breezy.tests.per_branch', 'breezy.tests.per_controldir', 'breezy.tests.per_controldir_colo', 'breezy.tests.per_foreign_vcs', 'breezy.tests.per_interrepository', 'breezy.tests.per_intertree', 'breezy.tests.per_interbranch', 'breezy.tests.per_lock', 'breezy.tests.per_merger', 'breezy.tests.per_transport', 'breezy.tests.per_tree', 'breezy.tests.per_repository', 'breezy.tests.per_repository_reference', 'breezy.tests.per_uifactory', 'breezy.tests.per_workingtree', 'breezy.tests.test__annotator', 'breezy.tests.test__known_graph', 'breezy.tests.test__walkdirs_win32', 'breezy.tests.test_ancestry', 'breezy.tests.test_annotate', 'breezy.tests.test_atomicfile', 'breezy.tests.test_bad_files', 'breezy.tests.test_bisect', 'breezy.tests.test_bisect_multi', 'breezy.tests.test_branch', 'breezy.tests.test_branchbuilder', 'breezy.tests.test_bugtracker', 'breezy.tests.test__chunks_to_lines', 'breezy.tests.test_cache_utf8', 'breezy.tests.test_chunk_writer', 'breezy.tests.test_clean_tree', 'breezy.tests.test_cmdline', 'breezy.tests.test_commands', 'breezy.tests.test_commit', 'breezy.tests.test_commit_merge', 'breezy.tests.test_config', 'breezy.tests.test_bedding', 'breezy.tests.test_conflicts', 'breezy.tests.test_controldir', 'breezy.tests.test_counted_lock', 'breezy.tests.test_crash', 'breezy.tests.test_decorators', 'breezy.tests.test_delta', 'breezy.tests.test_debug', 'breezy.tests.test_diff', 'breezy.tests.test_directory_service', 'breezy.tests.test_dirty_tracker', 'breezy.tests.test_email_message', 'breezy.tests.test_eol_filters', 'breezy.tests.test_errors', 'breezy.tests.test_estimate_compressed_size', 'breezy.tests.test_export', 'breezy.tests.test_export_pot', 'breezy.tests.test_extract', 'breezy.tests.test_features', 'breezy.tests.test_fetch', 'breezy.tests.test_fetch_ghosts', 'breezy.tests.test_fixtures', 'breezy.tests.test_fifo_cache', 'breezy.tests.test_filters', 'breezy.tests.test_filter_tree', 'breezy.tests.test_foreign', 'breezy.tests.test_forge', 'breezy.tests.test_generate_docs', 'breezy.tests.test_globbing', 'breezy.tests.test_gpg', 'breezy.tests.test_graph', 'breezy.tests.test_grep', 'breezy.tests.test_help', 'breezy.tests.test_hooks', 'breezy.tests.test_http', 'breezy.tests.test_http_response', 'breezy.tests.test_https_ca_bundle', 'breezy.tests.test_https_urllib', 'breezy.tests.test_i18n', 'breezy.tests.test_identitymap', 'breezy.tests.test_ignores', 'breezy.tests.test_import_tariff', 'breezy.tests.test_info', 'breezy.tests.test_lazy_import', 'breezy.tests.test_lazy_regex', 'breezy.tests.test_library_state', 'breezy.tests.test_location', 'breezy.tests.test_lock', 'breezy.tests.test_lockable_files', 'breezy.tests.test_lockdir', 'breezy.tests.test_log', 'breezy.tests.test_lru_cache', 'breezy.tests.test_lsprof', 'breezy.tests.test_mail_client', 'breezy.tests.test_matchers', 'breezy.tests.test_memorybranch', 'breezy.tests.test_memorytree', 'breezy.tests.test_merge', 'breezy.tests.test_mergeable', 'breezy.tests.test_merge_core', 'breezy.tests.test_merge_directive', 'breezy.tests.test_mergetools', 'breezy.tests.test_missing', 'breezy.tests.test_msgeditor', 'breezy.tests.test_multiparent', 'breezy.tests.test_multiwalker', 'breezy.tests.test_mutabletree', 'breezy.tests.test_nonascii', 'breezy.tests.test_options', 'breezy.tests.test_osutils', 'breezy.tests.test_osutils_encodings', 'breezy.tests.test_patch', 'breezy.tests.test_patches', 'breezy.tests.test_permissions', 'breezy.tests.test_plugins', 'breezy.tests.test_progress', 'breezy.tests.test_pyutils', 'breezy.tests.test_reconcile', 'breezy.tests.test_reconfigure', 'breezy.tests.test_registry', 'breezy.tests.test_rename_map', 'breezy.tests.test_revert', 'breezy.tests.test_revision', 'breezy.tests.test_revisionspec', 'breezy.tests.test_revisiontree', 'breezy.tests.test_rules', 'breezy.tests.test_url_policy_open', 'breezy.tests.test_sampler', 'breezy.tests.test_scenarios', 'breezy.tests.test_script', 'breezy.tests.test_selftest', 'breezy.tests.test_setup', 'breezy.tests.test_sftp_transport', 'breezy.tests.test_shelf', 'breezy.tests.test_shelf_ui', 'breezy.tests.test_smart_add', 'breezy.tests.test_smtp_connection', 'breezy.tests.test_source', 'breezy.tests.test_ssh_transport', 'breezy.tests.test_status', 'breezy.tests.test_strace', 'breezy.tests.test_subsume', 'breezy.tests.test_switch', 'breezy.tests.test_symbol_versioning', 'breezy.tests.test_tag', 'breezy.tests.test_test_server', 'breezy.tests.test_textfile', 'breezy.tests.test_textmerge', 'breezy.tests.test_cethread', 'breezy.tests.test_timestamp', 'breezy.tests.test_trace', 'breezy.tests.test_transactions', 'breezy.tests.test_transform', 'breezy.tests.test_transport', 'breezy.tests.test_transport_log', 'breezy.tests.test_tree', 'breezy.tests.test_treebuilder', 'breezy.tests.test_treeshape', 'breezy.tests.test_tsort', 'breezy.tests.test_ui', 'breezy.tests.test_uncommit', 'breezy.tests.test_upgrade', 'breezy.tests.test_upgrade_stacked', 'breezy.tests.test_upstream_import', 'breezy.tests.test_urlutils', 'breezy.tests.test_utextwrap', 'breezy.tests.test_version', 'breezy.tests.test_version_info', 'breezy.tests.test_views', 'breezy.tests.test_whitebox', 'breezy.tests.test_win32utils', 'breezy.tests.test_workspace', 'breezy.tests.test_workingtree', 'breezy.tests.test_wsgi']