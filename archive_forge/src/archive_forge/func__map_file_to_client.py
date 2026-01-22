from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_constants import DebugInfoHolder, IS_WINDOWS, IS_JYTHON, \
from _pydev_bundle._pydev_filesystem_encoding import getfilesystemencoding
from _pydevd_bundle.pydevd_comm_constants import file_system_encoding, filesystem_encoding_is_utf8
from _pydev_bundle.pydev_log import error_once
import json
import os.path
import sys
import itertools
import ntpath
from functools import partial
def _map_file_to_client(filename, cache=norm_filename_to_client_container):
    try:
        return cache[filename]
    except KeyError:
        abs_path = absolute_path(filename)
        translated_proper_case = get_path_with_real_case(abs_path)
        translated_normalized = normcase(abs_path)
        path_mapping_applied = False
        if translated_normalized.lower() != translated_proper_case.lower():
            if DEBUG_CLIENT_SERVER_TRANSLATION:
                pydev_log.critical('pydev debugger: translated_normalized changed path (from: %s to %s)', translated_proper_case, translated_normalized)
        for i, (eclipse_prefix, python_prefix) in enumerate(paths_from_eclipse_to_python):
            if translated_normalized.startswith(python_prefix):
                if DEBUG_CLIENT_SERVER_TRANSLATION:
                    pydev_log.critical('pydev debugger: replacing to client: %s', translated_normalized)
                eclipse_prefix = initial_paths[i][0]
                translated = eclipse_prefix + translated_proper_case[len(python_prefix):]
                if DEBUG_CLIENT_SERVER_TRANSLATION:
                    pydev_log.critical('pydev debugger: sent to client: %s - matched prefix: %s', translated, python_prefix)
                path_mapping_applied = True
                break
        else:
            if DEBUG_CLIENT_SERVER_TRANSLATION:
                pydev_log.critical('pydev debugger: to client: unable to find matching prefix for: %s in %s', translated_normalized, [x[1] for x in paths_from_eclipse_to_python])
            translated = translated_proper_case
        if eclipse_sep != python_sep:
            translated = translated.replace(python_sep, eclipse_sep)
        translated = _path_to_expected_str(translated)
        cache[filename] = (translated, path_mapping_applied)
        if translated not in _client_filename_in_utf8_to_source_reference:
            if path_mapping_applied:
                source_reference = 0
            else:
                source_reference = _next_source_reference()
                pydev_log.debug('Created source reference: %s for untranslated path: %s', source_reference, filename)
            _client_filename_in_utf8_to_source_reference[translated] = source_reference
            _source_reference_to_server_filename[source_reference] = filename
        return (translated, path_mapping_applied)