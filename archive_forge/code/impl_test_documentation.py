import duet
import duet.impl as impl
Make a task from the given future.

    We advance the task once, which just starts the generator and yields the
    future itself.
    