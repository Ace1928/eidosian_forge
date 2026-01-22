from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import subprocess
from googlecloudsdk.command_lib.ml_engine import local_predict
from googlecloudsdk.command_lib.ml_engine import predict_utilities
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
def RunPredict(model_dir, json_request=None, json_instances=None, text_instances=None, framework='tensorflow', signature_name=None):
    """Run ML Engine local prediction."""
    instances = predict_utilities.ReadInstancesFromArgs(json_request, json_instances, text_instances)
    sdk_root = config.Paths().sdk_root
    if not sdk_root:
        raise LocalPredictEnvironmentError('You must be running an installed Cloud SDK to perform local prediction.')
    env = os.environ.copy()
    encoding.SetEncodedValue(env, 'CLOUDSDK_ROOT', sdk_root)
    python_executables = files.SearchForExecutableOnPath('python')
    orig_py_path = encoding.GetEncodedValue(env, 'PYTHONPATH') or ''
    if orig_py_path:
        orig_py_path = ':' + orig_py_path
    encoding.SetEncodedValue(env, 'PYTHONPATH', os.path.join(sdk_root, 'lib', 'third_party', 'ml_sdk') + orig_py_path)
    if not python_executables:
        raise LocalPredictEnvironmentError("Something has gone really wrong; we can't find a valid Python executable on your PATH.")
    python_executable = properties.VALUES.ml_engine.local_python.Get() or python_executables[0]
    predict_args = ['--model-dir', model_dir, '--framework', framework]
    if signature_name:
        predict_args += ['--signature-name', signature_name]
    args = [encoding.Encode(a) for a in [python_executable, local_predict.__file__] + predict_args]
    proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    for instance in instances:
        proc.stdin.write((json.dumps(instance) + '\n').encode('utf-8'))
    proc.stdin.flush()
    output, err = proc.communicate()
    if proc.returncode != 0:
        raise LocalPredictRuntimeError(err)
    if err:
        log.warning(err)
    try:
        return json.loads(encoding.Decode(output))
    except ValueError:
        raise InvalidReturnValueError('The output for prediction is not in JSON format: ' + output)