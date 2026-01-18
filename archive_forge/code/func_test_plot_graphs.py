import unittest
import os
import numpy as np
from skimage import data, img_as_float
from pygsp import graphs, plotting
def test_plot_graphs(self):
    """
        Plot all graphs which have coordinates.
        With and without signal.
        With both backends.
        """
    COORDS_NO = {'Graph', 'BarabasiAlbert', 'ErdosRenyi', 'FullConnected', 'RandomRegular', 'StochasticBlockModel'}
    COORDS_WRONG_DIM = {'ImgPatches'}
    Gs = []
    for classname in set(graphs.__all__) - COORDS_NO - COORDS_WRONG_DIM:
        Graph = getattr(graphs, classname)
        if classname == 'NNGraph':
            Xin = np.arange(90).reshape(30, 3)
            Gs.append(Graph(Xin))
        elif classname in ['ImgPatches', 'Grid2dImgPatches']:
            Gs.append(Graph(img=self._img, patch_shape=(3, 3)))
        else:
            Gs.append(Graph())
        if classname == 'TwoMoons':
            Gs.append(Graph(moontype='standard'))
            Gs.append(Graph(moontype='synthesized'))
        elif classname == 'Cube':
            Gs.append(Graph(nb_dim=2))
            Gs.append(Graph(nb_dim=3))
        elif classname == 'DavidSensorNet':
            Gs.append(Graph(N=64))
            Gs.append(Graph(N=500))
            Gs.append(Graph(N=128))
    for G in Gs:
        self.assertTrue(hasattr(G, 'coords'))
        self.assertTrue(hasattr(G, 'A'))
        self.assertEqual(G.N, G.coords.shape[0])
        signal = np.arange(G.N) + 0.3
        G.plot(backend='pyqtgraph')
        G.plot(backend='matplotlib')
        G.plot_signal(signal, backend='pyqtgraph')
        G.plot_signal(signal, backend='matplotlib')
        plotting.close_all()