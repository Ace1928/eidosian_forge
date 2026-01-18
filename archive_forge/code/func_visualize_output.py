import numpy as np
import matplotlib.pyplot as plt
import ctypes
import sys
import argparse
import random
def visualize_output(layer_outputs, num_networks_to_view):
    contrasting_colors = ['b', 'g', 'r', 'c', 'm', 'y']
    central_node_color = 'k'
    central_node_weight = 1.0
    other_node_weight = 0.25
    for layer_idx, layer_output in enumerate(layer_outputs):
        if len(layer_output) >= num_networks_to_view:
            for network_idx in range(num_networks_to_view):
                plt.figure()
                network_output = layer_output[network_idx]
                for node_idx, node_output in enumerate(zip(*network_output)):
                    color = central_node_color if node_idx == 6 else contrasting_colors[node_idx % len(contrasting_colors)]
                    linewidth = central_node_weight if node_idx == 6 else other_node_weight
                    plt.plot(node_output, color=color, linewidth=linewidth)
                plt.title(f'Layer {layer_idx + 1}, Mini-Network {network_idx + 1}')
                plt.xlabel('Time Steps')
                plt.ylabel('Output Signal')
                plt.show()