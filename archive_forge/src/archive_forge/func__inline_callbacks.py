from . import version
import collections
from functools import wraps
import sys
import warnings
def _inline_callbacks(result, gen, deferred):
    """
    See inlineCallbacks.
    """
    waiting = [True, None]
    while 1:
        try:
            is_failure = isinstance(result, DeferredException)
            if is_failure:
                if sys.version_info[0] < 3:
                    result = gen.throw(result.type, result.value, result.traceback)
                else:
                    result = gen.throw(result.type(result.value).with_traceback(result.traceback))
            else:
                result = gen.send(result)
        except StopIteration:
            deferred.callback(None)
            return deferred
        except _DefGen_Return as err:
            appCodeTrace = sys.exc_info()[2].tb_next
            if is_failure:
                appCodeTrace = appCodeTrace.tb_next
            if appCodeTrace.tb_next and appCodeTrace.tb_next.tb_next:
                ultimateTrace = appCodeTrace
                while ultimateTrace.tb_next.tb_next:
                    ultimateTrace = ultimateTrace.tb_next
                filename = ultimateTrace.tb_frame.f_code.co_filename
                lineno = ultimateTrace.tb_lineno
                warnings.warn_explicit('returnValue() in %r causing %r to exit: returnValue should only be invoked by functions decorated with inlineCallbacks' % (ultimateTrace.tb_frame.f_code.co_name, appCodeTrace.tb_frame.f_code.co_name), DeprecationWarning, filename, lineno)
            deferred.callback(err.value)
            return deferred
        except:
            deferred.errback()
            return deferred
        if isinstance(result, Deferred):

            def gotResult(res):
                if waiting[0]:
                    waiting[0] = False
                    waiting[1] = res
                else:
                    _inline_callbacks(res, gen, deferred)
            result.add_callbacks(gotResult, gotResult)
            if waiting[0]:
                waiting[0] = False
                return deferred
            result = waiting[1]
            waiting[0] = True
            waiting[1] = None
    return deferred