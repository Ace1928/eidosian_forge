import importlib.metadata
import platform
import sys
def _print_info_dict(info_dict):
    """Print the information dictionary"""
    for key, stat in info_dict.items():
        print(f'{key:>10}: {stat}')