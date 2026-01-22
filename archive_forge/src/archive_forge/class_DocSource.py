from tensorflow.python.util import tf_export
class DocSource(object):
    """Specifies docstring source for a module.

  Only one of docstring or docstring_module_name should be set.
  * If docstring is set, then we will use this docstring when
    for the module.
  * If docstring_module_name is set, then we will copy the docstring
    from docstring source module.
  """

    def __init__(self, docstring=None, docstring_module_name=None):
        self.docstring = docstring
        self.docstring_module_name = docstring_module_name
        if self.docstring is not None and self.docstring_module_name is not None:
            raise ValueError('Only one of `docstring` or `docstring_module_name` can be set.')