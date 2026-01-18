import asyncio
import cProfile
import hashlib
import io
import itertools
import json
import logging
import resource
import os
import pstats
import queue
import re
import sys
import threading
import time
import traceback
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from difflib import SequenceMatcher
from functools import reduce
from logging.handlers import MemoryHandler, RotatingFileHandler
from logging import StreamHandler, FileHandler
from operator import itemgetter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional, Iterator, Union, Callable
import coloredlogs
import matplotlib.pyplot as plt
import mplcursors
import networkx as nx
import nltk
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import (
from wordcloud import WordCloud
from tqdm import tqdm
import requests
from transformers import BertModel, BertTokenizer
import functools
def visualize_data(self, data: pd.DataFrame, x_axis: str, y_axis: str, chart_type: str='scatter', z_axis: Optional[str]=None, color: Optional[str]=None, style: Optional[str]=None, interactive: bool=True, save_path: Optional[str]=None) -> None:
    """
        Visualize the processed data using the specified chart type and customization options.

        Args:
            data (pd.DataFrame): The processed data to visualize.
            x_axis (str): The column name to use for the x-axis.
            y_axis (str): The column name to use for the y-axis.
            chart_type (str): The type of chart to create ('scatter', 'line', 'bar', 'heatmap', '3d').
            z_axis (Optional[str]): The column name to use for the z-axis (for 3D plots).
            color (Optional[str]): The column name to use for color encoding.
            style (Optional[str]): The style theme for the chart.
            interactive (bool): Whether to enable interactive plot controls.
            save_path (Optional[str]): The path to save the visualization, if desired.

        Returns:
            None
        """
    try:
        self._validate_data(data, x_axis, y_axis, z_axis, color)
        self._validate_options(chart_type, style, interactive)
        x_data, y_data, z_data, color_data = self._prepare_data(data, x_axis, y_axis, z_axis, color)
        chart_creator = {'scatter': self._create_scatter_plot, 'line': self._create_line_plot, 'bar': self._create_bar_chart, 'heatmap': self._create_heatmap, '3d': self._create_3d_plot}
        if chart_type in chart_creator:
            chart_creator[chart_type](x_data, y_data, z_data, color_data, style, interactive)
        else:
            raise ValueError(f'Unsupported chart type: {chart_type}')
        plt.title(f'{chart_type.capitalize()} Plot of {x_axis} vs {y_axis}')
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        if save_path:
            self._save_visualization(save_path)
        plt.show()
    except Exception as e:
        self.logger.exception(f'An error occurred while visualizing data: {e}')
        advanced_logger.log(logging.ERROR, f'Data visualization failed: {e}')
        raise