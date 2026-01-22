from google.protobuf.internal import enum_type_wrapper
from google.protobuf.internal import python_message
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
def BuildServices(file_des, module_name, module):
    """Builds services classes and services stub class.

  Args:
    file_des: FileDescriptor of the .proto file
    module_name: str, the name of generated _pb2 module
    module: Generated _pb2 module
  """
    from google.protobuf import service as _service
    from google.protobuf import service_reflection
    for name, service in file_des.services_by_name.items():
        module[name] = service_reflection.GeneratedServiceType(name, (_service.Service,), dict(DESCRIPTOR=service, __module__=module_name))
        stub_name = name + '_Stub'
        module[stub_name] = service_reflection.GeneratedServiceStubType(stub_name, (module[name],), dict(DESCRIPTOR=service, __module__=module_name))