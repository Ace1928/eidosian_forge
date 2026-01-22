import math
import os
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_outputs import (
from ...modeling_utils import (
from ...utils import logging
from .configuration_markuplm import MarkupLMConfig
class XPathEmbeddings(nn.Module):
    """Construct the embeddings from xpath tags and subscripts.

    We drop tree-id in this version, as its info can be covered by xpath.
    """

    def __init__(self, config):
        super(XPathEmbeddings, self).__init__()
        self.max_depth = config.max_depth
        self.xpath_unitseq2_embeddings = nn.Linear(config.xpath_unit_hidden_size * self.max_depth, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = nn.ReLU()
        self.xpath_unitseq2_inner = nn.Linear(config.xpath_unit_hidden_size * self.max_depth, 4 * config.hidden_size)
        self.inner2emb = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.xpath_tag_sub_embeddings = nn.ModuleList([nn.Embedding(config.max_xpath_tag_unit_embeddings, config.xpath_unit_hidden_size) for _ in range(self.max_depth)])
        self.xpath_subs_sub_embeddings = nn.ModuleList([nn.Embedding(config.max_xpath_subs_unit_embeddings, config.xpath_unit_hidden_size) for _ in range(self.max_depth)])

    def forward(self, xpath_tags_seq=None, xpath_subs_seq=None):
        xpath_tags_embeddings = []
        xpath_subs_embeddings = []
        for i in range(self.max_depth):
            xpath_tags_embeddings.append(self.xpath_tag_sub_embeddings[i](xpath_tags_seq[:, :, i]))
            xpath_subs_embeddings.append(self.xpath_subs_sub_embeddings[i](xpath_subs_seq[:, :, i]))
        xpath_tags_embeddings = torch.cat(xpath_tags_embeddings, dim=-1)
        xpath_subs_embeddings = torch.cat(xpath_subs_embeddings, dim=-1)
        xpath_embeddings = xpath_tags_embeddings + xpath_subs_embeddings
        xpath_embeddings = self.inner2emb(self.dropout(self.activation(self.xpath_unitseq2_inner(xpath_embeddings))))
        return xpath_embeddings