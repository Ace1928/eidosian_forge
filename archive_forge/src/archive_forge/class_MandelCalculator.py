import logging
import math
import os
import sys
from taskflow import engines
from taskflow.engines.worker_based import worker
from taskflow.patterns import unordered_flow as uf
from taskflow import task
from taskflow.utils import threading_utils
class MandelCalculator(task.Task):

    def execute(self, image_config, mandelbrot_config, chunk):
        """Returns the number of iterations before the computation "escapes".

        Given the real and imaginary parts of a complex number, determine if it
        is a candidate for membership in the mandelbrot set given a fixed
        number of iterations.
        """

        def mandelbrot(x, y, max_iters):
            c = complex(x, y)
            z = 0j
            for i in range(max_iters):
                z = z * z + c
                if z.real * z.real + z.imag * z.imag >= 4:
                    return i
            return max_iters
        min_x, max_x, min_y, max_y, max_iters = mandelbrot_config
        height, width = image_config['size']
        pixel_size_x = (max_x - min_x) / width
        pixel_size_y = (max_y - min_y) / height
        block = []
        for y in range(chunk[0], chunk[1]):
            row = []
            imag = min_y + y * pixel_size_y
            for x in range(0, width):
                real = min_x + x * pixel_size_x
                row.append(mandelbrot(real, imag, max_iters))
            block.append(row)
        return block