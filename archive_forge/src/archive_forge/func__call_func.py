import functools
import traceback
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
def _call_func(self, args, kwargs):
    try:
        vars_at_start = self._template_store.variables()
        trainable_at_start = self._template_store.trainable_variables()
        if self._variables_created:
            result = self._func(*args, **kwargs)
        else:
            with trackable_util.capture_dependencies(template=self):
                result = self._func(*args, **kwargs)
        if self._variables_created:
            trainable_variables = self._template_store.trainable_variables()
            if len(trainable_at_start) != len(trainable_variables):
                raise ValueError('Trainable variable created when calling a template after the first time, perhaps you used tf.Variable when you meant tf.get_variable: %s' % list(object_identity.ObjectIdentitySet(trainable_variables) - object_identity.ObjectIdentitySet(trainable_at_start)))
            variables = self._template_store.variables()
            if len(vars_at_start) != len(variables):
                logging.info('New variables created when calling a template after the first time, perhaps you used tf.Variable when you meant tf.get_variable: %s', list(object_identity.ObjectIdentitySet(variables) - object_identity.ObjectIdentitySet(vars_at_start)))
        else:
            self._variables_created = True
        return result
    except Exception as exc:
        args = exc.args
        if not args:
            arg0 = ''
        else:
            arg0 = args[0]
        trace = ''.join(_skip_common_stack_elements(self._stacktrace, traceback.format_stack()))
        arg0 = '%s\n\noriginally defined at:\n%s' % (arg0, trace)
        new_args = [arg0]
        new_args.extend(args[1:])
        exc.args = tuple(new_args)
        raise