import re
import copy
import inspect
import ast
import textwrap
class ExpandVarargTransformer(ast.NodeTransformer):
    """
    Node transformer that replaces the starred use of a variable in an AST
    with a collection of unstarred named variables.
    """

    def __init__(self, starred_name, expand_names, *args, **kwargs):
        """
        Parameters
        ----------
        starred_name: str
            The name of the starred variable to replace
        expand_names: list of stf
            List of the new names that should be used to replace the starred
            variable

        """
        super(ExpandVarargTransformer, self).__init__(*args, **kwargs)
        self.starred_name = starred_name
        self.expand_names = expand_names