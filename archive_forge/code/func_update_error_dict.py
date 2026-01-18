import operator
from django.utils.hashable import make_hashable
def update_error_dict(self, error_dict):
    if hasattr(self, 'error_dict'):
        for field, error_list in self.error_dict.items():
            error_dict.setdefault(field, []).extend(error_list)
    else:
        error_dict.setdefault(NON_FIELD_ERRORS, []).extend(self.error_list)
    return error_dict