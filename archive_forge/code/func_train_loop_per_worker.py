import argparse
import os
import torch
import torch.nn.functional as F
from filelock import FileLock
from torch_geometric.datasets import FakeDataset, Reddit
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.transforms import RandomNodeSplit
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
def train_loop_per_worker(train_loop_config):
    dataset = train_loop_config['dataset_fn']()
    batch_size = train_loop_config['batch_size']
    num_epochs = train_loop_config['num_epochs']
    data = dataset[0]
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_idx = train_idx.split(train_idx.size(0) // train.get_context().get_world_size())[train.get_context().get_world_rank()]
    train_loader = NeighborSampler(data.edge_index, node_idx=train_idx, sizes=[25, 10], batch_size=batch_size, shuffle=True)
    train_loader = train.torch.prepare_data_loader(train_loader, add_dist_sampler=False)
    if train.get_context().get_world_rank() == 0:
        subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1], batch_size=2048, shuffle=False)
        subgraph_loader = train.torch.prepare_data_loader(subgraph_loader, add_dist_sampler=False)
    model = SAGE(dataset.num_features, 256, dataset.num_classes)
    model = train.torch.prepare_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x, y = (data.x.to(train.torch.get_device()), data.y.to(train.torch.get_device()))
    for epoch in range(num_epochs):
        model.train()
        for batch_size, n_id, adjs in train_loader:
            optimizer.zero_grad()
            out = model(x[n_id], adjs)
            loss = F.nll_loss(out, y[n_id[:batch_size]])
            loss.backward()
            optimizer.step()
        if train.get_context().get_world_rank() == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        train_accuracy = validation_accuracy = test_accuracy = None
        if train.get_context().get_world_rank() == 0:
            model.eval()
            with torch.no_grad():
                out = model.module.test(x, subgraph_loader)
            res = out.argmax(dim=-1) == data.y
            train_accuracy = int(res[data.train_mask].sum()) / int(data.train_mask.sum())
            validation_accuracy = int(res[data.val_mask].sum()) / int(data.val_mask.sum())
            test_accuracy = int(res[data.test_mask].sum()) / int(data.test_mask.sum())
        train.report(dict(train_accuracy=train_accuracy, validation_accuracy=validation_accuracy, test_accuracy=test_accuracy))