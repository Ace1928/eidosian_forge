from typing import Optional
def _set_chain(chain):
    """
    The function is used to set the chain by the users of
    the library at a global level. This global state then
    can be referenced by the library to get the chain instance.
    """
    globals()['__databricks_rag_chain__'] = chain