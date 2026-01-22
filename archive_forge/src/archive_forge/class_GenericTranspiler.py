import inspect
import threading
import types
import gast
from tensorflow.python.autograph.pyct import cache
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.pyct import loader
from tensorflow.python.autograph.pyct import naming
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.utils import ag_logging as logging
class GenericTranspiler(object):
    """A generic transpiler for Python functions.

  Its interface is the `transform` API, which can process Python function
  objects. Internally, it handles parsing.

  Users typically subclass this, customizing the `transform_ast` method. The
  output of transformed_ast is returned directly by `transform`. Existing
  methods like `transform_function` may also be overloaded.

  Example:

      class MyTransformer(GenericTranspiler):

        def transform_ast(self, node, ctx):
          result = <<transform node>>
          return result

      transformer = MyTransfomer()

      result = transformer.transform(f, ...)
      # result is the output
  """

    def get_transformed_name(self, node):
        """Returns a name for the output function. Subclasses may override this."""
        if isinstance(node, gast.Lambda):
            return 'lam'
        elif isinstance(node, gast.FunctionDef):
            return node.name
        raise ValueError('Unknown node type {}'.format(node))

    def transform_ast(self, node, ctx):
        """Performs an actual transformation of a function's AST.

    Subclasses must implement this method, and do not usually call it.

    Args:
      node: One or more ast.AST nodes representing the AST to be transformed.
      ctx: transformer.Context.
    """
        raise NotImplementedError('subclasses must override this')

    def transform(self, obj, user_context):
        """Transforms a Python object.

    Users typically call this method.

    Args:
      obj: A Python object, function, type, etc.
      user_context: An opaque object (may be None) that is forwarded to
        transform_ast, through the ctx.user attribute.
    Returns:
      The result of calling transform_function.

    Raises:
      NotImplementedError: if the type of obj is not handled.
    """
        if inspect.isfunction(obj) or inspect.ismethod(obj):
            return self.transform_function(obj, user_context)
        raise NotImplementedError('Non-function: {}'.format(type(obj)))

    def _erase_arg_defaults(self, node):
        """Erase arg default expressions, which would otherwise be unbound."""
        args = node.args
        for i in range(len(args.defaults)):
            args.defaults[i] = parser.parse_expression('None')
        for i, d in enumerate(args.kw_defaults):
            if d is not None:
                args.kw_defaults[i] = parser.parse_expression('None')
        return node

    def transform_module(self, mod, user_context):
        """Transforms a module.

    Subclasses may override this method. The return value is opaque.

    The method receives the original AST. The result is passed as-is to the
    output of `transform`.

    Args:
      mod: A Python module.
      user_context: An opaque object (may be None) that is forwarded to
        transform_ast, through the ctx.user attribute.
    Returns:
      List[Tuple[Any, Any]]. By default it returns the output of transform_ast,
      evaluated on each supported member, other than modules, together with a
      `transformer.Context` containing information about the transformation
      process.
    """
        result = []
        for member in mod.__dict__.values():
            if inspect.ismodule(member):
                continue
            try:
                result.append(self.transform(member, user_context))
            except NotImplementedError:
                pass
        return result

    def transform_function(self, fn, user_context):
        """Transforms a function.

    Subclasses may override this method. The return value is opaque.

    The method receives the original AST. The result is passed as-is to the
    output of `transform`.

    Args:
      fn: A function or lambda.
      user_context: An opaque object (may be None) that is forwarded to
        transform_ast, through the ctx.user attribute.
    Returns:
      Tuple[Any, Any]. By default it returns the output of transform_ast,
      together with a `transformer.Context` containing information about the
      transformation process.
    """
        future_features = inspect_utils.getfutureimports(fn)
        node, source = parser.parse_entity(fn, future_features=future_features)
        logging.log(3, 'Source code of %s:\n\n%s\n', fn, source)
        origin_info.resolve_entity(node, source, fn)
        namespace = inspect_utils.getnamespace(fn)
        namer = naming.Namer(namespace)
        new_name = namer.new_symbol(self.get_transformed_name(node), ())
        entity_info = transformer.EntityInfo(name=new_name, source_code=source, source_file='<fragment>', future_features=future_features, namespace=namespace)
        context = transformer.Context(entity_info, namer, user_context)
        node = self._erase_arg_defaults(node)
        result = self.transform_ast(node, context)
        return (result, context)