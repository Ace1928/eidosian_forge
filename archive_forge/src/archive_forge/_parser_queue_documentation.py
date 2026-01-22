import queue
from threading import RLock
from ..parser import Parser

    Thread safe message queue with built in MIDI parser.

    This should be avaiable to other backend implementations and perhaps
    also in the public API, but the API needs a bit of review. (Ideally This
    would replace the parser.)

    q = ParserQueue()

    q.put(msg)
    q.put_bytes([0xf8, 0, 0])

    msg = q.get()
    msg = q.poll()
    