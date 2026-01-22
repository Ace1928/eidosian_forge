from google.protobuf.internal import enum_type_wrapper
from google.protobuf.internal import python_message
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
def BuildMessage(msg_des):
    create_dict = {}
    for name, nested_msg in msg_des.nested_types_by_name.items():
        create_dict[name] = BuildMessage(nested_msg)
    create_dict['DESCRIPTOR'] = msg_des
    create_dict['__module__'] = module_name
    message_class = _reflection.GeneratedProtocolMessageType(msg_des.name, (_message.Message,), create_dict)
    _sym_db.RegisterMessage(message_class)
    return message_class