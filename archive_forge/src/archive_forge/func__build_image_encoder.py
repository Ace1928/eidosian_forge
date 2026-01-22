import torch
from torch import nn
from torch import optim
from parlai.agents.transformer.modules import TransformerEncoder
from parlai.agents.transformer import transformer as Transformer
def _build_image_encoder(self, n_layers_img):
    """
        Build the image encoder mapping raw image features to the appropriate space.

        :param n_layers_img:
            number of feed-forward layers for the image encoder
        """
    image_layers = [nn.BatchNorm1d(self.opt['image_features_dim']), nn.Dropout(p=self.opt['dropout']), nn.Linear(self.opt['image_features_dim'], self.opt['hidden_dim'])]
    for _ in range(n_layers_img - 1):
        image_layers += [nn.ReLU(), nn.Dropout(p=self.opt['dropout']), nn.Linear(self.opt['hidden_dim'], self.opt['hidden_dim'])]
    self.image_encoder = nn.Sequential(*image_layers)