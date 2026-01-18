import gc
def test_weak_method_func():
    from kivy.weakmethod import WeakMethod

    def do_something():
        pass
    weak_method = WeakMethod(do_something)
    assert not weak_method.is_dead()
    assert weak_method() == do_something
    assert weak_method == WeakMethod(do_something)
    del do_something
    gc.collect()
    assert not weak_method.is_dead()
    assert weak_method() is not None