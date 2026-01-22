import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging
import re
import os
import json
from collections import defaultdict, deque
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import math
import tensorboard
from torch.utils.tensorboard import SummaryWriter as writer
class INDEGO:

    def __init__(self, vocab_size, max_length, num_layers, num_heads, model, d_ff, dropout_rate, inputs, labels):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.model = model
        INDEGO.inputs = inputs
        INDEGO.labels = labels
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.embedding = self._initialize_embedding()
        self.layers = [TransformerLayer(model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)]

    def _initialize_embedding(self):
        return np.random.randn(self.vocab_size, self.d_model)

    async def forward(self, input_ids):
        embeddings = [self.embedding[token_id] for token_id in input_ids]
        for layer in self.layers:
            embeddings = await layer.forward(embeddings)
        return embeddings

    def compute_gradients(loss, model):
        gradients = {}
        grad_embedding = np.zeros_like(model.embedding)
        for i in range(len(model.embedding)):
            model.embedding[i] += 0.0001
            outputs = model.forward(INDEGO.inputs)
            logits = np.dot(outputs, model.embedding.T)
            loss_perturbed = nn.cross_entropy_loss(logits, INDEGO.labels)
            grad_embedding[i] = (loss_perturbed - loss) / 0.0001
            model.embedding[i] -= 0.0001
        gradients['embedding'] = grad_embedding
        for i, layer in enumerate(model.layers):
            grad_query_weights = np.zeros_like(layer.attention.query_weights)
            grad_key_weights = np.zeros_like(layer.attention.key_weights)
            grad_value_weights = np.zeros_like(layer.attention.value_weights)
            grad_output_weights = np.zeros_like(layer.attention.output_weights)
            for j in range(len(layer.attention.query_weights)):
                layer.attention.query_weights[j] += 0.0001
                outputs = model.forward(INDEGO.inputs)
                logits = np.dot(outputs, model.embedding.T)
                loss_perturbed = nn.cross_entropy_loss(logits, INDEGO.labels)
                grad_query_weights[j] = (loss_perturbed - loss) / 0.0001
                layer.attention.query_weights[j] -= 0.0001
                layer.attention.key_weights[j] += 0.0001
                outputs = model.forward(INDEGO.inputs)
                logits = np.dot(outputs, model.embedding.T)
                loss_perturbed = nn.cross_entropy_loss(logits, INDEGO.labels)
                grad_key_weights[j] = (loss_perturbed - loss) / 0.0001
                layer.attention.key_weights[j] -= 0.0001
                layer.attention.value_weights[j] += 0.0001
                outputs = model.forward(INDEGO.inputs)
                logits = np.dot(outputs, model.embedding.T)
                loss_perturbed = nn.cross_entropy_loss(logits, INDEGO.labels)
                grad_value_weights[j] = (loss_perturbed - loss) / 0.0001
                layer.attention.value_weights[j] -= 0.0001
                layer.attention.output_weights[j] += 0.0001
                outputs = model.forward(INDEGO.inputs)
                logits = np.dot(outputs, model.embedding.T)
                loss_perturbed = nn.cross_entropy_loss(logits, INDEGO.labels)
                grad_output_weights[j] = (loss_perturbed - loss) / 0.0001
                layer.attention.output_weights[j] -= 0.0001
            gradients[f'layer_{i}_attention_query_weights'] = grad_query_weights
            gradients[f'layer_{i}_attention_key_weights'] = grad_key_weights
            gradients[f'layer_{i}_attention_value_weights'] = grad_value_weights
            gradients[f'layer_{i}_attention_output_weights'] = grad_output_weights
            grad_dense1_weights = np.zeros_like(layer.feed_forward.dense1)
            grad_dense2_weights = np.zeros_like(layer.feed_forward.dense2)
            for j in range(len(layer.feed_forward.dense1)):
                layer.feed_forward.dense1[j] += 0.0001
                outputs = model.forward(INDEGO.inputs)
                logits = np.dot(outputs, model.embedding.T)
                loss_perturbed = nn.cross_entropy_loss(logits, INDEGO.labels)
                grad_dense1_weights[j] = (loss_perturbed - loss) / 0.0001
                layer.feed_forward.dense1[j] -= 0.0001
                layer.feed_forward.dense2[j] += 0.0001
                outputs = model.forward(INDEGO.inputs)
                logits = np.dot(outputs, model.embedding.T)
                loss_perturbed = nn.cross_entropy_loss(logits, INDEGO.labels)
                grad_dense2_weights[j] = (loss_perturbed - loss) / 0.0001
                layer.feed_forward.dense2[j] -= 0.0001
            gradients[f'layer_{i}_feed_forward_dense1'] = grad_dense1_weights
            gradients[f'layer_{i}_feed_forward_dense2'] = grad_dense2_weights
            grad_gamma1 = np.zeros_like(layer.layer_norm1.gamma)
            grad_beta1 = np.zeros_like(layer.layer_norm1.beta)
            grad_gamma2 = np.zeros_like(layer.layer_norm2.gamma)
            grad_beta2 = np.zeros_like(layer.layer_norm2.beta)
            for j in range(len(layer.layer_norm1.gamma)):
                layer.layer_norm1.gamma[j] += 0.0001
                outputs = model.forward(INDEGO.inputs)
                logits = np.dot(outputs, model.embedding.T)
                loss_perturbed = nn.cross_entropy_loss(logits, INDEGO.labels)
                grad_gamma1[j] = (loss_perturbed - loss) / 0.0001
                layer.layer_norm1.gamma[j] -= 0.0001
                layer.layer_norm1.beta[j] += 0.0001
                outputs = model.forward(INDEGO.inputs)
                logits = np.dot(outputs, model.embedding.T)
                loss_perturbed = nn.cross_entropy_loss(logits, INDEGO.labels)
                grad_beta1[j] = (loss_perturbed - loss) / 0.0001
                layer.layer_norm1.beta[j] -= 0.0001
                layer.layer_norm2.gamma[j] += 0.0001
                outputs = model.forward(INDEGO.inputs)
                logits = np.dot(outputs, model.embedding.T)
                loss_perturbed = nn.cross_entropy_loss(logits, INDEGO.labels)
                grad_gamma2[j] = (loss_perturbed - loss) / 0.0001
                layer.layer_norm2.gamma[j] -= 0.0001
                layer.layer_norm2.beta[j] += 0.0001
                outputs = model.forward(INDEGO.inputs)
                logits = np.dot(outputs, model.embedding.T)
                loss_perturbed = nn.cross_entropy_loss(logits, INDEGO.labels)
                grad_beta2[j] = (loss_perturbed - loss) / 0.0001
                layer.layer_norm2.beta[j] -= 0.0001
            gradients[f'layer_{i}_layer_norm1_gamma'] = grad_gamma1
            gradients[f'layer_{i}_layer_norm1_beta'] = grad_beta1
            gradients[f'layer_{i}_layer_norm2_gamma'] = grad_gamma2
            gradients[f'layer_{i}_layer_norm2_beta'] = grad_beta2
        return gradients

    async def train_step(model, inputs, labels, optimizer):
        outputs = await model.forward(inputs)
        loss = nn.compute_loss(outputs, labels)
        gradients = INDEGO.compute_gradients(loss, model)
        await optimizer.update_weights(model, gradients)
        return loss

    async def train(model, train_data, optimizer, num_epochs):
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_data:
                inputs, labels = batch
                loss = await INDEGO.train_step(model, inputs, labels, optimizer)
                total_loss += loss
            avg_loss = total_loss / len(train_data)
            logging.info(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')