import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.graphics.texture import Texture
import numpy as np
import matplotlib.pyplot as plt
import io
def update_fractal(self, instance):
    system_params = {}
    fractal_params = {}
    for key, value in self.param_inputs.items():
        try:
            if key in ['theta', 'f', 'g']:
                fractal_params[key] = float(value.text)
            else:
                system_params[key] = float(value.text)
        except ValueError:
            self.show_error(f'Invalid input for {key}')
            return
    x_range = np.linspace(-10, 10, 500)
    t_range = np.linspace(0, 10, 100)
    X, T = np.meshgrid(x_range, t_range)
    fractal_values = fractal_growth(X, T, system_params, fractal_params)
    self.plot_fractal_to_texture(fractal_values)