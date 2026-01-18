import os
import numpy as np
import pandas as pd
from statsmodels.api import add_constant
from statsmodels.genmod.tests.results import glm_test_resids

    # From R
    setwd('c:/workspace')
    data <- read.csv('fair.csv', sep=",")

    library(statmod)
    library(tweedie)

    model <- glm(affairs ~ rate_marriage + age + yrs_married -1, data=data,
             family=tweedie(var.power=1.5, link.power = 0))
    r <- resid(model, type='response')
    paste(as.character(r[1:17]), collapse=",")
    r <- resid(model, type='deviance')
    paste(as.character(r[1:17]), collapse=",")
    r <- resid(model, type='pearson')
    paste(as.character(r[1:17]), collapse=",")
    r <- resid(model, type='working')
    paste(as.character(r[1:17]), collapse=",")
    paste(as.character(model$coefficients[1:17]), collapse=",")
    s <- summary(model)
    paste(as.character(sqrt(diag(s$cov.scaled))), collapse=",")
    s$deviance
    paste(as.character(model$fitted.values[1:17]), collapse=",")
    