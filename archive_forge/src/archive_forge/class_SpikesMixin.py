import numpy as np
from ..core import Dataset, Dimension, util
from ..element import Bars, Graph
from ..element.util import categorical_aggregate2d
from .util import get_axis_padding
class SpikesMixin:

    def _get_axis_dims(self, element):
        if 'spike_length' in self.lookup_options(element, 'plot').options:
            return [element.dimensions()[0], None, None]
        return super()._get_axis_dims(element)

    def get_extents(self, element, ranges, range_type='combined', **kwargs):
        opts = self.lookup_options(element, 'plot').options
        if len(element.dimensions()) > 1 and 'spike_length' not in opts:
            ydim = element.get_dimension(1)
            s0, s1 = ranges[ydim.name]['soft']
            s0 = min(s0, 0) if util.isfinite(s0) else 0
            s1 = max(s1, 0) if util.isfinite(s1) else 0
            ranges[ydim.name]['soft'] = (s0, s1)
        proxy_dim = None
        if 'spike_length' in opts or len(element.dimensions()) == 1:
            proxy_dim = Dimension('proxy_dim')
            length = opts.get('spike_length', self.spike_length)
            if self.batched:
                bs, ts = ([], [])
                frame = self.current_frame or self.hmap.last
                for el in frame.values():
                    opts = self.lookup_options(el, 'plot').options
                    pos = opts.get('position', self.position)
                    bs.append(pos)
                    ts.append(pos + length)
                proxy_range = (np.nanmin(bs), np.nanmax(ts))
            else:
                proxy_range = (self.position, self.position + length)
            ranges['proxy_dim'] = {'data': proxy_range, 'hard': (np.nan, np.nan), 'soft': proxy_range, 'combined': proxy_range}
        return super().get_extents(element, ranges, range_type, ydim=proxy_dim)