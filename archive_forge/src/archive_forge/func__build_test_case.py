from __future__ import (absolute_import, division, print_function)
import os
import time
import re
from ansible import constants as C
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.plugins.callback import CallbackBase
from ansible.utils._junit_xml import (
def _build_test_case(self, task_data, host_data):
    """ build a TestCase from the given TaskData and HostData """
    name = '[%s] %s: %s' % (host_data.name, task_data.play, task_data.name)
    duration = host_data.finish - task_data.start
    if self._task_relative_path and task_data.path:
        junit_classname = to_text(os.path.relpath(to_bytes(task_data.path), to_bytes(self._task_relative_path)))
    else:
        junit_classname = task_data.path
    if self._replace_out_of_tree_path is not None and junit_classname.startswith('../'):
        junit_classname = self._replace_out_of_tree_path + to_text(os.path.basename(to_bytes(junit_classname)))
    if self._task_class == 'true':
        junit_classname = re.sub('\\.yml:[0-9]+$', '', junit_classname)
    if host_data.status == 'included':
        return TestCase(name=name, classname=junit_classname, time=duration, system_out=str(host_data.result))
    res = host_data.result._result
    rc = res.get('rc', 0)
    dump = self._dump_results(res, indent=0)
    dump = self._cleanse_string(dump)
    if host_data.status == 'ok':
        return TestCase(name=name, classname=junit_classname, time=duration, system_out=dump)
    test_case = TestCase(name=name, classname=junit_classname, time=duration)
    if host_data.status == 'failed':
        if 'exception' in res:
            message = res['exception'].strip().split('\n')[-1]
            output = res['exception']
            test_case.errors.append(TestError(message=message, output=output))
        elif 'msg' in res:
            message = res['msg']
            test_case.failures.append(TestFailure(message=message, output=dump))
        else:
            test_case.failures.append(TestFailure(message='rc=%s' % rc, output=dump))
    elif host_data.status == 'skipped':
        if 'skip_reason' in res:
            message = res['skip_reason']
        else:
            message = 'skipped'
        test_case.skipped = message
    return test_case