import os
from ....context import cpu
from ....initializer import Xavier
from ...block import HybridBlock
from ... import nn
from .... import base
def _make_features(self, layers, filters, batch_norm):
    featurizer = nn.HybridSequential(prefix='')
    for i, num in enumerate(layers):
        for _ in range(num):
            featurizer.add(nn.Conv2D(filters[i], kernel_size=3, padding=1, weight_initializer=Xavier(rnd_type='gaussian', factor_type='out', magnitude=2), bias_initializer='zeros'))
            if batch_norm:
                featurizer.add(nn.BatchNorm())
            featurizer.add(nn.Activation('relu'))
        featurizer.add(nn.MaxPool2D(strides=2))
    return featurizer