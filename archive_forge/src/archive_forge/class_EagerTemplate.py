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
class EagerTemplate(Template):
    """Wrap a function to aid in variable sharing in Eager mode.

  Templates are functions that create variables the first time they are called
  and reuse them thereafter. See `make_template` for full documentation.

  Note: By default, the full variable scope is captured at the time of first
  call. If `create_scope_now` is passed as True to the constructor, the full
  scope will be captured there, but no variables will be created until the first
  call.
  """

    def __init__(self, name, func, create_scope_now=False, custom_getter=None, create_graph_function=False):
        """Creates a template for the given function.

    Args:
      name: A name for the scope created by this template. The name will be made
        unique by appending `_N` to the it (see how
        `tf.compat.v1.variable_scope` treats the `default_name` for details).
      func: The function to apply each time.
      create_scope_now: Whether to create the scope at Template construction
        time, rather than first call. Defaults to false. Creating the scope at
        construction time may be more convenient if the template is passed
        through much lower level code, and you want to be sure of the scope name
        without knowing exactly where it will be first called. If set to True,
        the scope will be created in the constructor, and all subsequent times
        in `__call__`, leading to a trailing numeral being added to the names of
        all created Tensors. If set to False, the scope will be created at the
        first call location.
      custom_getter: optional custom getter to pass to `variable_scope()`
      create_graph_function: When True, `func` will be executed as a graph
        function. Enabling this flag allows the caller to reap the performance
        benefits associated with executing graphs, at the cost of sacrificing
        debuggability; however, not all Python functions can be compiled into
        graph functions. See the documentation for `function.defun` for details.

    Raises:
      RuntimeError: if eager execution is not enabled.
    """
        if not context.executing_eagerly():
            raise RuntimeError('{} objects can only be used when eager execution is enabled, use tf.Template for graph construction'.format(type(self)))
        super(EagerTemplate, self).__init__(name, func, create_scope_now, None, custom_getter, create_graph_function)
        if self._variable_scope is not None:
            variable_scope_name = self._variable_scope.name
        else:
            variable_scope_name = None
        self._template_store = _EagerTemplateVariableStore(variable_scope_name)
        self._variable_scope_context_manager = None

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

    def __call__(self, *args, **kwargs):
        if self._variable_scope:
            if not self._variable_scope_context_manager:
                self._variable_scope_context_manager = variable_scope.variable_scope(self._variable_scope, reuse=variable_scope.AUTO_REUSE)
            with self._variable_scope_context_manager:
                with self._template_store.as_default():
                    return self._call_func(args, kwargs)
        else:
            with variable_scope.variable_scope(self._unique_name, self._name, custom_getter=self._custom_getter) as vs:
                self._variable_scope = vs
                self._template_store.set_variable_scope_name(vs.name)
                with self._template_store.as_default():
                    return self._call_func(args, kwargs)

    @property
    def variables(self):
        """Returns the list of variables created by the Template."""
        if not self._variables_created:
            return []
        return self._template_store.variables()

    @property
    def trainable_variables(self):
        """Returns the list of trainable variables created by the Template."""
        if not self._variables_created:
            return []
        return self._template_store.trainable_variables()

    @property
    def non_trainable_variables(self):
        """Returns the list of non-trainable variables created by the Template."""
        if not self._variables_created:
            return []
        return self._template_store.non_trainable_variables()

    @property
    def global_variables(self):
        """Returns the list of global variables created by the Template."""
        if not self._variables_created:
            return []
        return self.variables

    @property
    def local_variables(self):
        """Returns the list of global variables created by the Template."""
        return []