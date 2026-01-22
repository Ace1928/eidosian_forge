import io
import math
import os
import typing
import weakref
def _show_fz_text(text):
    num_spans = 0
    num_chars = 0
    span = text.m_internal.head
    while 1:
        if not span:
            break
        num_spans += 1
        num_chars += span.len
        span = span.next
    return f'num_spans={num_spans} num_chars={num_chars}'