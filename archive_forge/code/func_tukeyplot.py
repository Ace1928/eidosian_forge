import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
def tukeyplot(results, dim=None, yticklabels=None):
    npairs = len(results)
    fig = plt.figure()
    fsp = fig.add_subplot(111)
    fsp.axis([-50, 50, 0.5, 10.5])
    fsp.set_title('95 % family-wise confidence level')
    fsp.title.set_y(1.025)
    fsp.set_yticks(np.arange(1, 11))
    fsp.set_yticklabels(['V-T', 'V-S', 'T-S', 'V-P', 'T-P', 'S-P', 'V-M', 'T-M', 'S-M', 'P-M'])
    fsp.yaxis.grid(True, linestyle='-', color='gray')
    fsp.set_xlabel('Differences in mean levels of Var', labelpad=8)
    fsp.xaxis.tick_bottom()
    fsp.yaxis.tick_left()
    xticklines = fsp.get_xticklines()
    for xtickline in xticklines:
        xtickline.set_marker(lines.TICKDOWN)
        xtickline.set_markersize(10)
    xlabels = fsp.get_xticklabels()
    for xlabel in xlabels:
        xlabel.set_y(-0.04)
    yticklines = fsp.get_yticklines()
    for ytickline in yticklines:
        ytickline.set_marker(lines.TICKLEFT)
        ytickline.set_markersize(10)
    ylabels = fsp.get_yticklabels()
    for ylabel in ylabels:
        ylabel.set_x(-0.04)
    for pair in range(npairs):
        data = 0.5 + results[pair] / 100.0
        fsp.axhline(y=npairs - pair, xmin=data.mean(), xmax=data[1], linewidth=1.25, color='blue', marker='|', markevery=1)
        fsp.axhline(y=npairs - pair, xmin=data[0], xmax=data.mean(), linewidth=1.25, color='blue', marker='|', markevery=1)
    fsp.axvline(x=0, linestyle='--', color='black')
    fig.subplots_adjust(bottom=0.125)