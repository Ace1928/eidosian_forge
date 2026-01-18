import enum
import sys
from tensorflow.python.util import tf_inspect
Recursively enumerate all members of `root`.

  Similar to the Python library function `os.path.walk`.

  Traverses the tree of Python objects starting with `root`, depth first.
  Parent-child relationships in the tree are defined by membership in modules or
  classes. The function `visit` is called with arguments
  `(path, parent, children)` for each module or class `parent` found in the tree
  of python objects starting with `root`. `path` is a string containing the name
  with which `parent` is reachable from the current context. For example, if
  `root` is a local class called `X` which contains a class `Y`, `visit` will be
  called with `('Y', X.Y, children)`).

  If `root` is not a module or class, `visit` is never called. `traverse`
  never descends into built-in modules.

  `children`, a list of `(name, object)` pairs are determined by
  `tf_inspect.getmembers`. To avoid visiting parts of the tree, `children` can
  be modified in place, using `del` or slice assignment.

  Cycles (determined by reference equality, `is`) stop the traversal. A stack of
  objects is kept to find cycles. Objects forming cycles may appear in
  `children`, but `visit` will not be called with any object as `parent` which
  is already in the stack.

  Traversing system modules can take a long time, it is advisable to pass a
  `visit` callable which denylists such modules.

  Args:
    root: A python object with which to start the traversal.
    visit: A function taking arguments `(path, parent, children)`. Will be
      called for each object found in the traversal.
  