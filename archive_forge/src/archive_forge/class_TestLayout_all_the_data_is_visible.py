import pytest
class TestLayout_all_the_data_is_visible:

    def compute_layout(self, *, n_cols, n_rows, orientation, n_data, clock):
        """Returns {view-index: pos, view-index: pos, ...}"""
        from textwrap import dedent
        from kivy.lang import Builder
        rv = Builder.load_string(dedent(f"\n            RecycleView:\n                viewclass: 'Widget'\n                size: 300, 300\n                data: ({{}} for __ in range({n_data}))\n                RecycleGridLayout:\n                    id: layout\n                    cols: {n_cols}\n                    rows: {n_rows}\n                    orientation: '{orientation}'\n                    default_size_hint: None, None\n                    default_size: 100, 100\n                    size_hint: None, None\n                    size: self.minimum_size\n            "))
        clock.tick()
        layout = rv.ids.layout
        return {layout.get_view_index_at(c.center): tuple(c.pos) for c in layout.children}

    @pytest.mark.parametrize('n_cols, n_rows', [(1, None), (None, 1), (1, 1)])
    def test_1x1(self, kivy_clock, n_cols, n_rows):
        from kivy.uix.recyclegridlayout import RecycleGridLayout
        for orientation in RecycleGridLayout.orientation.options:
            assert {0: (0, 0)} == self.compute_layout(n_data=1, orientation=orientation, n_cols=n_cols, n_rows=n_rows, clock=kivy_clock)

    @pytest.mark.parametrize('n_cols, n_rows', [(3, None), (None, 1), (3, 1)])
    @pytest.mark.parametrize('orientation', 'lr-tb lr-bt tb-lr bt-lr'.split())
    def test_3x1_lr(self, kivy_clock, orientation, n_cols, n_rows):
        assert {0: (0, 0), 1: (100, 0), 2: (200, 0)} == self.compute_layout(n_data=3, orientation=orientation, n_cols=n_cols, n_rows=n_rows, clock=kivy_clock)

    @pytest.mark.parametrize('n_cols, n_rows', [(3, None), (None, 1), (3, 1)])
    @pytest.mark.parametrize('orientation', 'rl-tb rl-bt tb-rl bt-rl'.split())
    def test_3x1_rl(self, kivy_clock, orientation, n_cols, n_rows):
        assert {0: (200, 0), 1: (100, 0), 2: (0, 0)} == self.compute_layout(n_data=3, orientation=orientation, n_cols=n_cols, n_rows=n_rows, clock=kivy_clock)

    @pytest.mark.parametrize('n_cols, n_rows', [(1, None), (None, 3), (1, 3)])
    @pytest.mark.parametrize('orientation', 'tb-lr tb-rl lr-tb rl-tb'.split())
    def test_1x3_tb(self, kivy_clock, orientation, n_cols, n_rows):
        assert {0: (0, 200), 1: (0, 100), 2: (0, 0)} == self.compute_layout(n_data=3, orientation=orientation, n_cols=n_cols, n_rows=n_rows, clock=kivy_clock)

    @pytest.mark.parametrize('n_cols, n_rows', [(1, None), (None, 3), (1, 3)])
    @pytest.mark.parametrize('orientation', 'bt-lr bt-rl lr-bt rl-bt'.split())
    def test_1x3_bt(self, kivy_clock, orientation, n_cols, n_rows):
        assert {0: (0, 0), 1: (0, 100), 2: (0, 200)} == self.compute_layout(n_data=3, orientation=orientation, n_cols=n_cols, n_rows=n_rows, clock=kivy_clock)

    @pytest.mark.parametrize('n_cols, n_rows', [(2, None), (None, 2), (2, 2)])
    def test_2x2_lr_tb(self, kivy_clock, n_cols, n_rows):
        assert {0: (0, 100), 1: (100, 100), 2: (0, 0)} == self.compute_layout(n_data=3, orientation='lr-tb', n_cols=n_cols, n_rows=n_rows, clock=kivy_clock)

    @pytest.mark.parametrize('n_cols, n_rows', [(2, None), (None, 2), (2, 2)])
    def test_2x2_lr_bt(self, kivy_clock, n_cols, n_rows):
        assert {0: (0, 0), 1: (100, 0), 2: (0, 100)} == self.compute_layout(n_data=3, orientation='lr-bt', n_cols=n_cols, n_rows=n_rows, clock=kivy_clock)

    @pytest.mark.parametrize('n_cols, n_rows', [(2, None), (None, 2), (2, 2)])
    def test_2x2_rl_tb(self, kivy_clock, n_cols, n_rows):
        assert {0: (100, 100), 1: (0, 100), 2: (100, 0)} == self.compute_layout(n_data=3, orientation='rl-tb', n_cols=n_cols, n_rows=n_rows, clock=kivy_clock)

    @pytest.mark.parametrize('n_cols, n_rows', [(2, None), (None, 2), (2, 2)])
    def test_2x2_rl_bt(self, kivy_clock, n_cols, n_rows):
        assert {0: (100, 0), 1: (0, 0), 2: (100, 100)} == self.compute_layout(n_data=3, orientation='rl-bt', n_cols=n_cols, n_rows=n_rows, clock=kivy_clock)

    @pytest.mark.parametrize('n_cols, n_rows', [(2, None), (None, 2), (2, 2)])
    def test_2x2_tb_lr(self, kivy_clock, n_cols, n_rows):
        assert {0: (0, 100), 1: (0, 0), 2: (100, 100)} == self.compute_layout(n_data=3, orientation='tb-lr', n_cols=n_cols, n_rows=n_rows, clock=kivy_clock)

    @pytest.mark.parametrize('n_cols, n_rows', [(2, None), (None, 2), (2, 2)])
    def test_2x2_tb_rl(self, kivy_clock, n_cols, n_rows):
        assert {0: (100, 100), 1: (100, 0), 2: (0, 100)} == self.compute_layout(n_data=3, orientation='tb-rl', n_cols=n_cols, n_rows=n_rows, clock=kivy_clock)

    @pytest.mark.parametrize('n_cols, n_rows', [(2, None), (None, 2), (2, 2)])
    def test_2x2_bt_lr(self, kivy_clock, n_cols, n_rows):
        assert {0: (0, 0), 1: (0, 100), 2: (100, 0)} == self.compute_layout(n_data=3, orientation='bt-lr', n_cols=n_cols, n_rows=n_rows, clock=kivy_clock)

    @pytest.mark.parametrize('n_cols, n_rows', [(2, None), (None, 2), (2, 2)])
    def test_2x2_bt_rl(self, kivy_clock, n_cols, n_rows):
        assert {0: (100, 0), 1: (100, 100), 2: (0, 0)} == self.compute_layout(n_data=3, orientation='bt-rl', n_cols=n_cols, n_rows=n_rows, clock=kivy_clock)