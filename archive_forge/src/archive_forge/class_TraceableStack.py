import inspect
class TraceableStack(object):
    """A stack of TraceableObjects."""

    def __init__(self, existing_stack=None):
        """Constructor.

    Args:
      existing_stack: [TraceableObject, ...] If provided, this object will
        set its new stack to a SHALLOW COPY of existing_stack.
    """
        self._stack = existing_stack[:] if existing_stack else []

    def push_obj(self, obj, offset=0):
        """Add object to the stack and record its filename and line information.

    Args:
      obj: An object to store on the stack.
      offset: Integer.  If 0, the caller's stack frame is used.  If 1,
          the caller's caller's stack frame is used.

    Returns:
      TraceableObject.SUCCESS if appropriate stack information was found,
      TraceableObject.HEURISTIC_USED if the stack was smaller than expected,
      and TraceableObject.FAILURE if the stack was empty.
    """
        traceable_obj = TraceableObject(obj)
        self._stack.append(traceable_obj)
        return traceable_obj.set_filename_and_line_from_caller(offset + 1)

    def pop_obj(self):
        """Remove last-inserted object and return it, without filename/line info."""
        return self._stack.pop().obj

    def peek_top_obj(self):
        """Return the most recent stored object."""
        return self._stack[-1].obj

    def peek_objs(self):
        """Return iterator over stored objects ordered newest to oldest."""
        return (t_obj.obj for t_obj in reversed(self._stack))

    def peek_traceable_objs(self):
        """Return iterator over stored TraceableObjects ordered newest to oldest."""
        return reversed(self._stack)

    def __len__(self):
        """Return number of items on the stack, and used for truth-value testing."""
        return len(self._stack)

    def copy(self):
        """Return a copy of self referencing the same objects but in a new list.

    This method is implemented to support thread-local stacks.

    Returns:
      TraceableStack with a new list that holds existing objects.
    """
        return TraceableStack(self._stack)