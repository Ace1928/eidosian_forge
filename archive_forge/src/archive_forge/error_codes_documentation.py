from FAILED_PRECONDITION.  INVALID_ARGUMENT indicates arguments

An extra enum entry to prevent people from writing code that
fails to compile when a new code is added.

Nobody should ever reference this enumeration entry. In particular,
if you write C++ code that switches on this enumeration, add a default:
case instead of a case that mentions this enumeration entry.

Nobody should rely on the value (currently 20) listed here.  It
may change in the future.
