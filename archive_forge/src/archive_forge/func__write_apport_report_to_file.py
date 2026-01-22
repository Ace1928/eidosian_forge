import os
import platform
import pprint
import sys
import time
from io import StringIO
import breezy
from . import bedding, debug, osutils, plugin, trace
def _write_apport_report_to_file(exc_info):
    import traceback
    from apport.report import Report
    exc_type, exc_object, exc_tb = exc_info
    pr = Report()
    pr.add_proc_info()
    del pr['ProcMaps']
    pr.add_user_info()
    pr['SourcePackage'] = 'brz'
    pr['Package'] = 'brz'
    pr['CommandLine'] = pprint.pformat(sys.argv)
    pr['BrzVersion'] = breezy.__version__
    pr['PythonVersion'] = breezy._format_version_tuple(sys.version_info)
    pr['Platform'] = platform.platform(aliased=1)
    pr['UserEncoding'] = osutils.get_user_encoding()
    pr['FileSystemEncoding'] = sys.getfilesystemencoding()
    pr['Locale'] = os.environ.get('LANG', 'C')
    pr['BrzPlugins'] = _format_plugin_list()
    pr['PythonLoadedModules'] = _format_module_list()
    pr['BrzDebugFlags'] = pprint.pformat(debug.debug_flags)
    pr['SourcePackage'] = 'brz'
    pr['Package'] = 'brz'
    pr['CrashDb'] = 'brz'
    tb_file = StringIO()
    traceback.print_exception(exc_type, exc_object, exc_tb, file=tb_file)
    pr['Traceback'] = tb_file.getvalue()
    _attach_log_tail(pr)
    pr.anonymize()
    if pr.check_ignored():
        return None
    else:
        crash_file_name, crash_file = _open_crash_file()
        pr.write(crash_file)
        crash_file.close()
        return crash_file_name