from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2Process(_messages.Message):
    """Represents an operating system process.

  Fields:
    args: Process arguments as JSON encoded strings.
    argumentsTruncated: True if `args` is incomplete.
    binary: File information for the process executable.
    envVariables: Process environment variables.
    envVariablesTruncated: True if `env_variables` is incomplete.
    libraries: File information for libraries loaded by the process.
    name: The process name, as displayed in utilities like `top` and `ps`.
      This name can be accessed through `/proc/[pid]/comm` and changed with
      `prctl(PR_SET_NAME)`.
    parentPid: The parent process ID.
    pid: The process ID.
    script: When the process represents the invocation of a script, `binary`
      provides information about the interpreter, while `script` provides
      information about the script file provided to the interpreter.
  """
    args = _messages.StringField(1, repeated=True)
    argumentsTruncated = _messages.BooleanField(2)
    binary = _messages.MessageField('GoogleCloudSecuritycenterV2File', 3)
    envVariables = _messages.MessageField('GoogleCloudSecuritycenterV2EnvironmentVariable', 4, repeated=True)
    envVariablesTruncated = _messages.BooleanField(5)
    libraries = _messages.MessageField('GoogleCloudSecuritycenterV2File', 6, repeated=True)
    name = _messages.StringField(7)
    parentPid = _messages.IntegerField(8)
    pid = _messages.IntegerField(9)
    script = _messages.MessageField('GoogleCloudSecuritycenterV2File', 10)