from io import BytesIO
from . import tree
from .filters import ContentFilterContext, filtered_output_bytes
Construct a new filtered tree view.

        :param filter_stack_callback: A callable taking a path that returns
            the filter stack that should be used for that path.
        :param backing_tree: An underlying tree to wrap.
        