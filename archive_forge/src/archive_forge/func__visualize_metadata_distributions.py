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
def _visualize_metadata_distributions(self, metadata: Dict[str, Any], style: Optional[str]=None, interactive: bool=True, save_path: Optional[str]=None) -> None:
    """
        Create a 3D scatter plot for the title similarity scores.

        Args:
            metadata (Dict[str, Any]): The metadata to visualize.
            style (Optional[str]): The style theme for the charts.
            interactive (bool): Whether to enable interactive plot controls.
            save_path (Optional[str]): The path to save the visualizations, if desired.

        Returns:
            None
        """
    if 'title_similarity_scores' in metadata:
        title_similarity_scores = pd.DataFrame(metadata['title_similarity_scores'])
        self._create_3d_plot(x_data=title_similarity_scores['doc_id_1'], y_data=title_similarity_scores['doc_id_2'], z_data=title_similarity_scores['similarity_score'], color_data=None, style=style, interactive=interactive)
        plt.title('Title Similarity Scores')
        plt.xlabel('Document ID 1')
        plt.ylabel('Document ID 2')
        plt.zlabel('Similarity Score')
        if save_path:
            save_path_title_similarity_scores = f'{save_path}_title_similarity_scores.png'
            self._save_visualization(save_path_title_similarity_scores)