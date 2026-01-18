import pytest
@pytest.mark.parametrize('transition_cls_name', transition_cls_names)
def test_switching_does_not_affect_a_list_of_screens(transition_cls_name):
    import kivy.uix.screenmanager as sm
    transition_cls = getattr(sm, transition_cls_name)
    scrmgr = sm.ScreenManager()
    for i in range(3):
        scrmgr.add_widget(sm.Screen(name=str(i)))
    names = list(scrmgr.screen_names)
    scrmgr.transition = transition_cls()
    scrmgr.current = '1'
    assert names == scrmgr.screen_names
    scrmgr.transition = transition_cls()
    scrmgr.current = '2'
    assert names == scrmgr.screen_names