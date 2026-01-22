import sys
import logging
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import OpenGL.GLU as glu
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Basic3DCube:
    """
    This class encapsulates a simple 3D cube rendered using OpenGL. It provides a foundational
    structure for understanding and manipulating basic 3D graphics with precision and detail.

    Attributes:
        width (int): The width of the window.
        height (int): The height of the window.
    """

    def __init__(self, width=800, height=600):
        """
        Initializes the Basic3DCube with a specified width and height.

        Args:
            width (int): The width of the window, default is 800.
            height (int): The height of the window, default is 600.
        """
        self.width = width
        self.height = height
        self.init_glut()

    def init_glut(self):
        """
        Initializes the GLUT environment for rendering the 3D cube.
        """
        try:
            glut.glutInit(sys.argv)
            glut.glutInitDisplayMode(
                glut.GLUT_DOUBLE | glut.GLUT_RGBA | glut.GLUT_DEPTH
            )
            glut.glutInitWindowSize(self.width, self.height)
            glut.glutCreateWindow(b"Basic 3D Cube")
            glut.glutDisplayFunc(self.display)
            glut.glutReshapeFunc(self.reshape)
            self.init_opengl()
        except Exception as e:
            logging.error("Failed to initialize GLUT environment: %s", e)
            raise

    def init_opengl(self):
        """
        Initializes OpenGL settings for rendering.
        """
        try:
            gl.glClearColor(0.0, 0.0, 0.0, 1.0)  # Set the background color to black
            gl.glEnable(gl.GL_DEPTH_TEST)  # Enable depth testing
        except Exception as e:
            logging.error("OpenGL initialization failed: %s", e)
            raise

    def display(self):
        """
        The display function called by GLUT to render the scene.
        """
        try:
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glLoadIdentity()
            glu.gluLookAt(3, 3, 3, 0, 0, 0, 0, 1, 0)  # Position the camera
            self.draw_cube()
            glut.glutSwapBuffers()
        except Exception as e:
            logging.error("Display function failed: %s", e)
            raise

    def draw_cube(self):
        """
        Draws a simple 3D cube using OpenGL.
        """
        try:
            vertices = [
                [1, 1, -1],
                [1, -1, -1],
                [-1, -1, -1],
                [-1, 1, -1],
                [1, 1, 1],
                [1, -1, 1],
                [-1, -1, 1],
                [-1, 1, 1],
            ]
            edges = [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),
            ]
            gl.glBegin(gl.GL_LINES)
            for edge in edges:
                for vertex in edge:
                    gl.glVertex3fv(vertices[vertex])
            gl.glEnd()
        except Exception as e:
            logging.error("Cube drawing failed: %s", e)
            raise

    def reshape(self, w, h):
        """
        Handles the window reshape event.
        """
        try:
            self.width, self.height = w, h
            gl.glViewport(0, 0, self.width, self.height)
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()
            glu.gluPerspective(45, self.width / self.height, 0.1, 100.0)
            gl.glMatrixMode(gl.GL_MODELVIEW)
        except Exception as e:
            logging.error("Reshape function failed: %s", e)
            raise


if __name__ == "__main__":
    try:
        cube = Basic3DCube()
        glut.glutMainLoop()
    except Exception as e:
        logging.error("Main execution failed: %s", e)
        raise
