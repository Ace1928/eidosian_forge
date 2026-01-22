import torch._C
from torch._C import _add_docstr as add_docstr
from_file(filename, shared=False, size=0) -> Storage
def add_docstr_all(method, docstr):
    for cls_name in storage_classes:
        cls = getattr(torch._C, cls_name)
        try:
            add_docstr(getattr(cls, method), docstr)
        except AttributeError:
            pass