import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect

    For someone who want to test by script. Comment it cause github ONNX CI
    do not have the torch python package.
    @staticmethod
    def export_gridsample_torch():  # type: () -> None
        node = onnx.helper.make_node(
            'GridSample',
            inputs=['X', 'Grid'],
            outputs=['Y'],
            mode='bilinear',
            padding_mode='zeros',
            align_corners=0,
        )

        # X shape, [N, C, H, W] - [1, 1, 4, 4]
        # Grid shape, [N, H_out, W_out, 2] - [1, 6, 6, 2]
        # Y shape, [N, C, H_out, W_out] - [1, 1, 6, 6]
        import torch
        X = torch.arange(3 * 3).view(1, 1, 3, 3).float()
        d = torch.linspace(-1, 1, 6)
        meshx, meshy = torch.meshgrid((d, d))
        grid = torch.stack((meshy, meshx), 2)
        Grid = grid.unsqueeze(0)
        Y = torch.nn.functional.grid_sample(X, Grid, mode='bilinear',
                                            padding_mode='zeros', align_corners=False)
        expect(node, inputs=[X.numpy(), Grid.numpy()], outputs=[Y.numpy()],
               name='test_gridsample_torch')
    