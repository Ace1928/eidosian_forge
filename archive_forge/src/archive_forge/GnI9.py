import pyopencl as cl
import numpy as np
import functools
import os
import logging
from typing import Any, Dict, Tuple
from collections import deque
import pickle
import shutil
import OpenGL.GL as gl

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

DEFAULT_SHADER_DIRECTORY = "/media/lloyd/Aurora_M2/extra-repos/pandas3D/shaders"


class ShaderManager:
    """
    A class meticulously designed to manage shader operations within a GPU context, specifically tailored to compile and manage shaders
    for rendering in a graphics and physics engine using OpenGL for shader operations and OpenCL for computational tasks. This manager supports a comprehensive range of shaders including vertex, fragment, geometry,
    tessellation control, tessellation evaluation, and compute shaders.

    Attributes:
        context (cl.Context): The OpenCL context associated with a specific device where computational tasks will be managed.
        shader_cache (Dict[Tuple[str, str], Any]): Cache to store compiled OpenGL shader programs to avoid recompilation, utilizing an LRU cache mechanism.
        pinned_memory_buffers (Dict[str, cl.Buffer]): A dictionary to manage pinned memory buffers for efficient host-to-device data transfer.
        default_shader_directory (str): The default directory to store and retrieve shader files.
        default_shaders (Dict[str, str]): Mapping of shader types to default shader filenames.
    """

    def __init__(
        self,
        context: cl.Context = None,
        default_shader_directory: str = DEFAULT_SHADER_DIRECTORY,
    ):
        """
        Initializes the ShaderManager with a given OpenCL context for computational tasks and a default directory for shaders.

        Parameters:
            context (cl.Context): The OpenCL context to be used for computational tasks.
            default_shader_directory (str): The default directory to store and retrieve shader files.
        """
        self.context = context if context else cl.create_some_context()
        self.shader_cache = {}  # Correct initialization
        self.pinned_memory_buffers = {}
        self.default_shader_directory = default_shader_directory
        self.default_shaders = {
            "vertex": "vertex_shader.glsl",
            "fragment": "fragment_shader.glsl",
            "geometry": "geometry_shader.glsl",
        }
        if not os.path.exists(self.default_shader_directory):
            os.makedirs(self.default_shader_directory)
            logging.info(
                f"Created default shader directory at {self.default_shader_directory}"
            )

    def load_shader_from_file(
        self, filename: str = "", shader_type: str = "vertex"
    ) -> Any:
        """
        Loads a shader from a file, using a default shader if the specified file is not found, and compiles it using OpenGL.

        Parameters:
            filename (str): The filename of the shader file.
            shader_type (str): The type of shader to load.

        Returns:
            Any: The compiled OpenGL shader program.
        """
        if not filename:
            filename = self.default_shaders.get(shader_type, "")
            if not filename:
                error_message = f"No default shader file for type: {shader_type}"
                logging.error(error_message)
                raise ValueError(error_message)

        shader_path = os.path.join(self.default_shader_directory, filename)

        try:
            with open(shader_path, "r") as file:
                shader_code = file.read()
            return self.compile_shader(shader_code, shader_type)
        except FileNotFoundError:
            logging.error(f"Shader file not found: {shader_path}")
            raise
        except Exception as e:
            logging.error(f"Error reading shader file {shader_path}: {str(e)}")
            raise

    def compile_shader(self, source: str, shader_type: str = "vertex") -> Any:
        """
        Compiles a shader from source code using OpenGL.

        Parameters:
            source (str): The shader source code.
            shader_type (str): The type of shader to compile.

        Returns:
            Any: The compiled OpenGL shader.
        """
        shader_ref = {
            "vertex": gl.GL_VERTEX_SHADER,
            "fragment": gl.GL_FRAGMENT_SHADER,
            "geometry": gl.GL_GEOMETRY_SHADER,
        }.get(shader_type, gl.GL_VERTEX_SHADER)

        shader = gl.glCreateShader(shader_ref)
        gl.glShaderSource(shader, source)
        gl.glCompileShader(shader)

        # Check for compile errors
        compile_success = gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS)
        if not compile_success:
            info_log = gl.glGetShaderInfoLog(shader)
            gl.glDeleteShader(shader)
            raise Exception(f"Error compiling {shader_type} shader: {info_log}")
        return shader
