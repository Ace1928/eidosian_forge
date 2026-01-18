import ray
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
def train_func(netD, netG, optimG, optimD, criterion, dataloader, iteration, device, mnist_model_ref):
    real_label = 1
    fake_label = 0
    for i, data in enumerate(dataloader, 0):
        if i >= train_iterations_per_step:
            break
        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimD.step()
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimG.step()
        is_score, is_std = inception_score(fake, mnist_model_ref)
        if iteration % 10 == 0:
            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f \tInception score: %.4f' % (iteration, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, is_score))
    return (errG.item(), errD.item(), is_score)