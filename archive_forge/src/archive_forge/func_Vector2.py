from .. import sage_helper
def Vector2(*args, **kwargs):
    ans = Vector(*args, **kwargs)
    assert len(ans) == 2
    return ans