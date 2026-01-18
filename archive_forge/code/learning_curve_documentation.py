from warnings import simplefilter
import numpy as np
from sklearn import model_selection
import wandb
from wandb.sklearn import utils
Train model on datasets of varying size and generates plot of score vs size.

    Called by plot_learning_curve to visualize learning curve. Please use the function
    plot_learning_curve() if you wish to visualize your learning curves.
    