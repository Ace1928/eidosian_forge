from dataclasses import fields
def add_dynamic_docstring(*docstr, text, dynamic_elements):

    def docstring_decorator(fn):
        func_doc = (fn.__doc__ or '') + ''.join(docstr)
        fn.__doc__ = func_doc + text.format(**dynamic_elements)
        return fn
    return docstring_decorator