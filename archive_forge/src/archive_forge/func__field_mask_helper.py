import collections
import collections.abc
import copy
import inspect
from cloudsdk.google.protobuf import field_mask_pb2
from cloudsdk.google.protobuf import message
from cloudsdk.google.protobuf import wrappers_pb2
def _field_mask_helper(original, modified, current=''):
    answer = []
    for name in original.DESCRIPTOR.fields_by_name:
        field_path = _get_path(current, name)
        original_val = getattr(original, name)
        modified_val = getattr(modified, name)
        if _is_message(original_val) or _is_message(modified_val):
            if original_val != modified_val:
                if _is_wrapper(original_val) or _is_wrapper(modified_val):
                    answer.append(field_path)
                elif not modified_val.ListFields():
                    answer.append(field_path)
                else:
                    answer.extend(_field_mask_helper(original_val, modified_val, field_path))
        elif original_val != modified_val:
            answer.append(field_path)
    return answer