from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_utils import hasattr_checked, DAPGrouper, Timer
from io import StringIO
import traceback
from os.path import basename
from functools import partial
from _pydevd_bundle.pydevd_constants import IS_PY36_OR_GREATER, \
from _pydevd_bundle.pydevd_safe_repr import SafeRepr
from _pydevd_bundle import pydevd_constants
class TupleResolver:

    def resolve(self, var, attribute):
        """
        :param var: that's the original object we're dealing with.
        :param attribute: that's the key to resolve
            -- either the dict key in get_dictionary or the name in the dap protocol.
        """
        if attribute in (GENERATED_LEN_ATTR_NAME, TOO_LARGE_ATTR):
            return None
        try:
            return var[int(attribute)]
        except:
            if attribute == 'more':
                return MoreItems(var, pydevd_constants.PYDEVD_CONTAINER_INITIAL_EXPANDED_ITEMS)
            return getattr(var, attribute)

    def get_contents_debug_adapter_protocol(self, lst, fmt=None):
        """
        This method is to be used in the case where the variables are all saved by its id (and as
        such don't need to have the `resolve` method called later on, so, keys don't need to
        embed the reference in the key).

        Note that the return should be ordered.

        :return list(tuple(name:str, value:object, evaluateName:str))
        """
        lst_len = len(lst)
        ret = []
        format_str = '%0' + str(int(len(str(lst_len - 1)))) + 'd'
        if fmt is not None and fmt.get('hex', False):
            format_str = '0x%0' + str(int(len(hex(lst_len).lstrip('0x')))) + 'x'
        initial_expanded = pydevd_constants.PYDEVD_CONTAINER_INITIAL_EXPANDED_ITEMS
        for i, item in enumerate(lst):
            ret.append((format_str % i, item, '[%s]' % i))
            if i >= initial_expanded - 1:
                if lst_len - initial_expanded < pydevd_constants.PYDEVD_CONTAINER_BUCKET_SIZE:
                    item = MoreItemsRange(lst, initial_expanded, lst_len)
                else:
                    item = MoreItems(lst, initial_expanded)
                ret.append(('more', item, None))
                break
        from_default_resolver = defaultResolver.get_contents_debug_adapter_protocol(lst, fmt=fmt)
        if from_default_resolver:
            ret = from_default_resolver + ret
        ret.append((GENERATED_LEN_ATTR_NAME, len(lst), partial(_apply_evaluate_name, evaluate_name='len(%s)')))
        return ret

    def get_dictionary(self, var, fmt={}):
        l = len(var)
        d = {}
        format_str = '%0' + str(int(len(str(l - 1)))) + 'd'
        if fmt is not None and fmt.get('hex', False):
            format_str = '0x%0' + str(int(len(hex(l).lstrip('0x')))) + 'x'
        initial_expanded = pydevd_constants.PYDEVD_CONTAINER_INITIAL_EXPANDED_ITEMS
        for i, item in enumerate(var):
            d[format_str % i] = item
            if i >= initial_expanded - 1:
                item = MoreItems(var, initial_expanded)
                d['more'] = item
                break
        additional_fields = defaultResolver.get_dictionary(var)
        d.update(additional_fields)
        d[GENERATED_LEN_ATTR_NAME] = len(var)
        return d