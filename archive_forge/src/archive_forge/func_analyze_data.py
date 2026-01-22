import sqlite3
import csv
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
import requests
def analyze_data(data):
    if not data:
        return
    mean_value = sum(data.values()) / len(data)
    return {'mean': mean_value}