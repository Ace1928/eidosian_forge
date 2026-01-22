from eventlet import event as _event
Suspend the caller only if our count is nonzero. In that case,
        resume the caller once the count decrements to zero again.
        