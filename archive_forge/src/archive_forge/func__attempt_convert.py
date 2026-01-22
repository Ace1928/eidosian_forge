from .api import NoDefault, Invalid
from .compound import CompoundValidator, from_python
def _attempt_convert(self, value, state, validate):
    if self.convert_to_list:
        value = self._convert_to_list(value)
    if self.if_empty is not NoDefault and (not value):
        return self.if_empty
    if self.not_empty and (not value):
        if validate is from_python and self.accept_python:
            return []
        raise Invalid(self.message('empty', state), value, state)
    new_list = []
    errors = []
    all_good = True
    is_set = isinstance(value, set)
    if state is not None:
        previous_index = getattr(state, 'index', NoDefault)
        previous_full_list = getattr(state, 'full_list', NoDefault)
        index = 0
        state.full_list = value
    try:
        for sub_value in value:
            if state:
                state.index = index
                index += 1
            good_pass = True
            for validator in self.validators:
                try:
                    sub_value = validate(validator, sub_value, state)
                except Invalid as e:
                    errors.append(e)
                    all_good = False
                    good_pass = False
                    break
            if good_pass:
                errors.append(None)
            new_list.append(sub_value)
        if all_good:
            if is_set:
                new_list = set(new_list)
            return new_list
        else:
            raise Invalid('Errors:\n%s' % '\n'.join((str(e) for e in errors if e)), value, state, error_list=errors)
    finally:
        if state is not None:
            if previous_index is NoDefault:
                try:
                    del state.index
                except AttributeError:
                    pass
            else:
                state.index = previous_index
            if previous_full_list is NoDefault:
                try:
                    del state.full_list
                except AttributeError:
                    pass
            else:
                state.full_list = previous_full_list