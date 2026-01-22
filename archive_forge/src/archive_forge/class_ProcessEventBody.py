from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class ProcessEventBody(BaseSchema):
    """
    "body" of ProcessEvent

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'name': {'type': 'string', 'description': "The logical name of the process. This is usually the full path to process's executable file. Example: /home/example/myproj/program.js."}, 'systemProcessId': {'type': 'integer', 'description': 'The system process id of the debugged process. This property will be missing for non-system processes.'}, 'isLocalProcess': {'type': 'boolean', 'description': 'If true, the process is running on the same computer as the debug adapter.'}, 'startMethod': {'type': 'string', 'enum': ['launch', 'attach', 'attachForSuspendedLaunch'], 'description': 'Describes how the debug engine started debugging this process.', 'enumDescriptions': ['Process was launched under the debugger.', 'Debugger attached to an existing process.', 'A project launcher component has launched a new process in a suspended state and then asked the debugger to attach.']}, 'pointerSize': {'type': 'integer', 'description': 'The size of a pointer or address for this process, in bits. This value may be used by clients when formatting addresses for display.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, name, systemProcessId=None, isLocalProcess=None, startMethod=None, pointerSize=None, update_ids_from_dap=False, **kwargs):
        """
        :param string name: The logical name of the process. This is usually the full path to process's executable file. Example: /home/example/myproj/program.js.
        :param integer systemProcessId: The system process id of the debugged process. This property will be missing for non-system processes.
        :param boolean isLocalProcess: If true, the process is running on the same computer as the debug adapter.
        :param string startMethod: Describes how the debug engine started debugging this process.
        :param integer pointerSize: The size of a pointer or address for this process, in bits. This value may be used by clients when formatting addresses for display.
        """
        self.name = name
        self.systemProcessId = systemProcessId
        self.isLocalProcess = isLocalProcess
        self.startMethod = startMethod
        self.pointerSize = pointerSize
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        name = self.name
        systemProcessId = self.systemProcessId
        isLocalProcess = self.isLocalProcess
        startMethod = self.startMethod
        pointerSize = self.pointerSize
        dct = {'name': name}
        if systemProcessId is not None:
            dct['systemProcessId'] = systemProcessId
        if isLocalProcess is not None:
            dct['isLocalProcess'] = isLocalProcess
        if startMethod is not None:
            dct['startMethod'] = startMethod
        if pointerSize is not None:
            dct['pointerSize'] = pointerSize
        dct.update(self.kwargs)
        return dct