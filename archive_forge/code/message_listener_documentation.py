Called every time the message is modified in such a way that the parent
    message may need to be updated.  This currently means either:
    (a) The message was modified for the first time, so the parent message
        should henceforth mark the message as present.
    (b) The message's cached byte size became dirty -- i.e. the message was
        modified for the first time after a previous call to ByteSize().
        Therefore the parent should also mark its byte size as dirty.
    Note that (a) implies (b), since new objects start out with a client cached
    size (zero).  However, we document (a) explicitly because it is important.

    Modified() will *only* be called in response to one of these two events --
    not every time the sub-message is modified.

    Note that if the listener's |dirty| attribute is true, then calling
    Modified at the moment would be a no-op, so it can be skipped.  Performance-
    sensitive callers should check this attribute directly before calling since
    it will be true most of the time.
    